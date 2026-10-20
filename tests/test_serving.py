"""Serving API: health, analysis outputs (no verdict), upload limits and rejection of
malformed input, in-memory handling, no content in logs, offline model loading, the
limitations text against the results files, and no train/serve skew on a real page."""
from __future__ import annotations

import copy
import csv
import io
import json
import logging
import tempfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docforge.analyze import analyze_page, mask_regions
from docforge.model import build_model, save_checkpoint
from docforge.scoring import binarize, top_k_mean
from research.config import resolve
from serving.app import Analyzer, BodySizeLimit, create_app, file_sha256, load_settings
from serving.limitations import CITED_NUMBERS, LIMITATIONS

VERDICT_WORDS = ("verdict", "tier", "is_tampered", "authentic", "tampered", "prediction", "decision")


@pytest.fixture(scope="module")
def tiny_model_files(tmp_path_factory):
    """Randomly initialised U-Net checkpoint + a copy of the frozen config with its hash."""
    d = tmp_path_factory.mktemp("serve")
    torch_seed = 0
    import torch
    torch.manual_seed(torch_seed)
    ckpt = d / "model.pt"
    save_checkpoint(ckpt, build_model(encoder_weights=None), epoch=0)
    frozen = json.loads(resolve("results/frozen_config_run2.json").read_text())
    frozen["checkpoint_sha256"] = file_sha256(ckpt)
    fz = d / "frozen.json"
    fz.write_text(json.dumps(frozen))
    return ckpt, fz


def make_settings(ckpt, fz, **upload):
    s = copy.deepcopy(load_settings())
    s["model"]["checkpoint"], s["model"]["frozen_config"] = str(ckpt), str(fz)
    s["upload"].update(upload)
    return s


@pytest.fixture(scope="module")
def client(tiny_model_files):
    with TestClient(create_app(make_settings(*tiny_model_files))) as c:
        yield c


def encode(img: np.ndarray, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, fmt, **kw)
    return buf.getvalue()


def page(h=300, w=240, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 240, np.uint8)
    img[40:60, 20:200] = rng.integers(0, 90, (20, 180, 3), dtype=np.uint8)
    return img


# --- health and analysis ------------------------------------------------------------------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True
    assert r.json()["settings"]["stride"] == 128 and r.json()["settings"]["pixel_threshold"] == 0.88


def test_index_page_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and "/v1/analyze" in r.text


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_analyze_returns_maps_boxes_score_and_no_verdict(client, fmt):
    img = page()
    r = client.post("/v1/analyze", files={"file": ("doc." + fmt.lower(), encode(img, fmt), "image/" + fmt.lower())})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["image"] == {"format": fmt, "width": 240, "height": 300}
    assert d["risk_score"]["label"] == "experimental" and 0.0 <= d["risk_score"]["value"] <= 1.0
    assert d["limitations"] == LIMITATIONS
    for key in ("heatmap_png_base64", "overlay_jpeg_base64", "regions", "n_regions", "predicted_area_fraction"):
        assert key in d
    assert all(w not in k for k in d for w in VERDICT_WORDS)          # no verdict/tier field anywhere at top level
    assert "image_threshold" not in json.dumps(d)                      # the unreliable operating point is not exposed
    assert r.headers["cache-control"] == "no-store"
    heat = Image.open(io.BytesIO(__import__("base64").b64decode(d["heatmap_png_base64"])))
    assert heat.size == (240, 300)


def test_limitations_text_matches_results_files():
    for text, path, keys in CITED_NUMBERS:
        node = json.loads(resolve(path).read_text())
        for k in keys:
            node = node[k]
        expected = f"{round(node * 100)}%" if text.endswith("%") else f"{node:.2f}"
        assert text == expected, (text, path, keys, node)
        assert text in LIMITATIONS
    for phrase in ("synthetic data only", "splices", "text overwrites", "blur", "not reliable as a screening filter"):
        assert phrase in LIMITATIONS
    assert "likely authentic" not in LIMITATIONS.lower() and "probability" not in LIMITATIONS.lower()


# --- upload rejection ----------------------------------------------------------------------
def test_rejects_file_over_size_limit(client):
    big = b"\xff\xd8" + bytes(10 * 1024 * 1024 + 10)
    r = client.post("/v1/analyze", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_rejects_over_limit_even_when_body_fits(tiny_model_files):
    """File-level limit (not just the body limit): a 60 kB PNG against a 50 kB limit."""
    rng = np.random.default_rng(1)
    data = encode(rng.integers(0, 255, (150, 150, 3), dtype=np.uint8), "PNG")
    assert len(data) > 50_000
    with TestClient(create_app(make_settings(*tiny_model_files, max_bytes=50_000))) as c:
        r = c.post("/v1/analyze", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 413 and r.json()["error"]["code"] == "file_too_large"


def test_body_limit_applies_to_streamed_uploads_without_length():
    sent = []

    async def app(scope, receive, send):
        while (await receive()).get("more_body"):
            pass

    async def receive_chunks():
        return {"type": "http.request", "body": b"x" * 600, "more_body": True}

    async def send(msg):
        sent.append(msg)

    import asyncio
    from starlette.exceptions import HTTPException
    mw = BodySizeLimit(app, max_body=1000)
    with pytest.raises(HTTPException) as e:
        asyncio.run(mw({"type": "http", "path": "/v1/analyze", "headers": []}, receive_chunks, send))
    assert e.value.status_code == 413


@pytest.mark.parametrize("name,data,status,code", [
    ("empty.png", b"", 400, "empty_file"),
    ("noise.jpg", bytes(np.random.default_rng(2).integers(0, 255, 5000, dtype=np.uint8)), 415, "unsupported_or_malformed"),
    ("text.png", b"hello, this is not an image" * 20, 415, "unsupported_or_malformed"),
])
def test_rejects_malformed_files(client, name, data, status, code):
    r = client.post("/v1/analyze", files={"file": (name, data, "image/png")})
    assert r.status_code == status and r.json()["error"]["code"] == code


def test_rejects_truncated_jpeg(client):
    data = encode(page(), "JPEG")
    r = client.post("/v1/analyze", files={"file": ("cut.jpg", data[: len(data) // 2], "image/jpeg")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "malformed_image"


def test_rejects_other_formats_even_with_image_mime_type(client):
    data = encode(page(), "GIF")
    r = client.post("/v1/analyze", files={"file": ("fake.png", data, "image/png")})
    assert r.status_code == 415 and r.json()["error"]["code"] == "unsupported_format"


@pytest.mark.parametrize("size,code", [((9000, 40), "image_too_large"), ((6400, 6400), "image_too_large"),
                                       ((20, 300), "image_too_small")])
def test_decoded_dimension_checks(client, size, code):
    data = encode(np.full((size[1], size[0]), 255, np.uint8), "PNG")  # tiny file, large header dimensions
    r = client.post("/v1/analyze", files={"file": ("d.png", data, "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == code


# --- privacy -------------------------------------------------------------------------------
def test_uploads_stay_in_memory(client, monkeypatch):
    """A 2 MB upload (above Starlette's default 1 MB spool size) must not roll over to disk."""
    def no_disk(self):
        raise AssertionError("upload was spooled to disk")
    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", no_disk)
    data = bytes(np.random.default_rng(3).integers(0, 255, 2 * 1024 * 1024, dtype=np.uint8))
    r = client.post("/v1/analyze", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 415  # parsed fully in memory, then rejected as not an image


def test_logs_contain_no_file_name_or_content(client, caplog):
    caplog.set_level(logging.DEBUG)
    client.post("/v1/analyze", files={"file": ("PATIENT_Jane_Doe_invoice.png", encode(page(), "PNG"), "image/png")})
    client.post("/v1/analyze", files={"file": ("PATIENT_Jane_Doe_bad.png", b"secret-bytes-123", "image/png")})
    assert "Jane_Doe" not in caplog.text and "secret-bytes" not in caplog.text
    assert "rid=" in caplog.text


# --- model loading -------------------------------------------------------------------------
def test_model_loads_offline(tiny_model_files, monkeypatch):
    import huggingface_hub
    import torch.hub

    def no_net(*a, **k):
        raise AssertionError("network download attempted")
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", no_net)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_net)
    an = Analyzer(make_settings(*tiny_model_files))
    an.load()
    assert an.ready, an.error


def test_checkpoint_hash_mismatch_refuses_to_serve(tiny_model_files, tmp_path):
    ckpt, fz = tiny_model_files
    frozen = json.loads(fz.read_text())
    frozen["checkpoint_sha256"] = "0" * 64
    bad = tmp_path / "frozen_bad.json"
    bad.write_text(json.dumps(frozen))
    with TestClient(create_app(make_settings(ckpt, bad))) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/v1/analyze", files={"file": ("a.png", encode(page(), "PNG"), "image/png")})
        assert r.status_code == 503


def test_missing_checkpoint_reports_unavailable(tiny_model_files, tmp_path):
    with TestClient(create_app(make_settings(tmp_path / "nope.pt", tiny_model_files[1]))) as c:
        r = c.get("/health")
    assert r.status_code == 503 and "not found" in r.json()["error"]


# --- docforge.analyze ----------------------------------------------------------------------
def test_regions_and_score_follow_frozen_definitions():
    prob = np.zeros((100, 120), np.float32)
    prob[10:20, 30:50] = 0.95
    prob[60:65, 5:8] = 0.90
    prob[80:90, 100:110] = 0.50            # below the 0.88 threshold
    frozen = {"pixel_threshold": 0.88, "n_bins": 1000, "score_rule": "top_k_mean", "top_k": 500}
    res = analyze_page(np.full((100, 120, 3), 200, np.uint8), prob, frozen, display_max_side=60)
    assert res["n_regions"] == 2
    assert res["regions"][0]["bbox_xyxy"] == [30, 10, 50, 20] and res["regions"][0]["area_px"] == 200
    assert res["regions"][1]["bbox_xyxy"] == [5, 60, 8, 65]
    assert res["risk_score"] == pytest.approx(top_k_mean(prob, 500))
    assert res["predicted_area_fraction"] == pytest.approx(binarize(prob, 0.88, 1000).mean())
    assert max(res["overlay"].shape[:2]) == 60 and res["heatmap"].shape == res["overlay"].shape
    assert mask_regions(np.zeros((5, 5), bool), np.zeros((5, 5)), 10) == ([], 0)


def test_serving_score_matches_evaluation_on_a_real_page():
    """No train/serve skew: the API's risk score for a validation page equals the score the
    evaluation harness recorded (results/val_model_run2_per_image.csv). Needs the gitignored
    checkpoint and the generated dataset; skipped otherwise."""
    ckpt, csv_path = resolve("artifacts/run2/best.pt"), resolve("results/val_model_run2_per_image.csv")
    man = resolve("data/manifests/val.jsonl")
    if not (ckpt.exists() and man.exists() and csv_path.exists()):
        pytest.skip("needs artifacts/run2/best.pt, data/ and results/val_model_run2_per_image.csv")
    rows = {r["sample_id"]: r for r in csv.DictReader(open(csv_path))}
    rec = next(json.loads(l) for l in open(man) if json.loads(l)["manipulation_type"] == "splice")
    settings = load_settings()
    settings["model"]["checkpoint"] = str(ckpt)
    an = Analyzer(settings)
    an.load()
    assert an.ready, an.error
    img = np.asarray(Image.open(resolve("data") / rec["image_path"]).convert("RGB"))
    with open(resolve("data") / rec["image_path"], "rb") as f:
        from serving.validation import decode_upload
        img2, _ = decode_upload(f.read(), settings["upload"])
    assert np.array_equal(img, img2)
    res = analyze_page(img2, an(img2), an.frozen)
    assert res["risk_score"] == pytest.approx(float(rows[rec["sample_id"]]["score"]), abs=1e-4)
