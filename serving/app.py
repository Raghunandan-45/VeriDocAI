"""VeriDoc AI local API: tampering-localisation heatmaps for a single document image.

    VERIDOC_CHECKPOINT=artifacts/run2/best.pt .venv/bin/uvicorn serving.app:app --port 8000
    # then open http://127.0.0.1:8000/  (upload page) or POST /v1/analyze

Design rules (from the Phase 4 test results):
- No tampered/authentic verdict and no risk tier. The response holds a heatmap, an overlay,
  boxes of predicted regions, an experimental risk score and a fixed limitations text.
- Inference uses the same docforge code and validation-frozen pixel threshold / score rule
  as the evaluation. The checkpoint must match the hash recorded in the frozen config.
- Uploads are processed in memory only: multipart files are kept off disk, nothing is
  saved, and logs record sizes and timings but never file names or image content.
- The model is built with encoder_weights=None and the trained weights are loaded from a
  local file, so serving needs no network access.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from starlette.formparsers import MultiPartParser

from docforge.analyze import analyze_page
from serving.limitations import LIMITATIONS
from serving.validation import UploadError, decode_upload

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger("veridoc.serving")

RISK_SCORE_NOTE = ("Experimental score: the mean of the 500 highest pixel scores. It is not a "
                   "calibrated probability, no threshold is applied to it, and it is not reliable "
                   "as a screening filter (see limitations).")


def _path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_settings(config_path: str | Path = "configs/serve.yaml") -> dict:
    """configs/serve.yaml plus the VERIDOC_* environment overrides."""
    cfg = yaml.safe_load(_path(config_path).read_text())
    cfg["model"]["checkpoint"] = os.environ.get("VERIDOC_CHECKPOINT", cfg["model"]["checkpoint"])
    cfg["model"]["frozen_config"] = os.environ.get("VERIDOC_FROZEN_CONFIG", cfg["model"]["frozen_config"])
    cfg["inference"]["stride"] = int(os.environ.get("VERIDOC_STRIDE", cfg["inference"]["stride"]))
    threads = os.environ.get("VERIDOC_TORCH_THREADS")
    cfg["inference"]["torch_threads"] = int(threads) if threads else None
    return cfg


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Analyzer:
    """Loads the checkpoint once and runs full-page inference, one page at a time (a lock
    keeps concurrent requests from oversubscribing the CPU threads)."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.ready, self.error, self.info = False, None, {}
        self._lock = threading.Lock()
        self.frozen = json.loads(_path(settings["model"]["frozen_config"]).read_text())
        self.tiling = yaml.safe_load(_path(settings["model"]["eval_config"]).read_text())
        self.stride = settings["inference"]["stride"]

    def load(self) -> None:
        import torch
        from docforge.model import load_checkpoint, make_predict_fn

        ckpt = _path(self.settings["model"]["checkpoint"])
        if not ckpt.is_file():
            self.error = "checkpoint file not found (set VERIDOC_CHECKPOINT)"
            return
        sha = file_sha256(ckpt)
        if sha != self.frozen.get("checkpoint_sha256"):
            self.error = "checkpoint sha256 does not match the frozen config; thresholds would not apply"
            return
        if self.settings["inference"]["torch_threads"]:
            torch.set_num_threads(self.settings["inference"]["torch_threads"])
        model, meta = load_checkpoint(ckpt, "cpu")
        self.predict_fn = make_predict_fn(model, "cpu")
        self.info = {"checkpoint_sha256": sha, "checkpoint_epoch": meta.get("epoch"),
                     "torch_threads": torch.get_num_threads()}
        self.ready = True

    def settings_summary(self) -> dict:
        t = self.tiling
        return {"pixel_threshold": self.frozen["pixel_threshold"], "score_rule": self.frozen["score_rule"],
                "top_k": self.frozen["top_k"], "tile": t["tiling"]["tile"], "stride": self.stride,
                "stride_used_in_test_evaluation": self.frozen["stride"],
                "analysis_max_side": t["analysis"]["max_side"], **self.info}

    def __call__(self, img: np.ndarray) -> np.ndarray:
        from docforge.model import predict_page

        t = self.tiling
        with self._lock:
            return predict_page(img, self.predict_fn, t["analysis"]["max_side"], t["tiling"]["tile"],
                                self.stride, t["tiling"]["batch_size"], t["tiling"]["window_floor"])


class BodySizeLimit:
    """ASGI middleware: reject request bodies above `max_body` bytes with 413, both from the
    Content-Length header and while streaming (chunked uploads without a length)."""

    def __init__(self, app, max_body: int, path_prefix: str = "/v1/"):
        self.app, self.max_body, self.prefix = app, max_body, path_prefix

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(self.prefix):
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        length = headers.get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.max_body:
            return await _send_json(send, 413, _error("file_too_large", "The request body is too large."))
        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_body:
                    from starlette.exceptions import HTTPException
                    raise HTTPException(413, "The request body is too large.")
            return message

        return await self.app(scope, limited_receive, send)


async def _send_json(send, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _b64(img: np.ndarray, fmt: str, quality: int = 85) -> str:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, fmt, **({"quality": quality} if fmt == "JPEG" else {"optimize": True}))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def create_app(settings: dict | None = None, analyzer: Analyzer | None = None) -> FastAPI:
    settings = settings or load_settings()
    up, outcfg = settings["upload"], settings["output"]
    # keep uploaded files in memory: Starlette spools multipart files to a temp file above
    # spool_max_size, so raise it above the largest body we accept. It is a process-wide class
    # attribute, so only ever raise it (another app in the same process may need more).
    MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size,
                                         up["max_bytes"] + up["multipart_overhead"] + 1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        an = app.state.analyzer
        if not an.ready and an.error is None:
            an.load()
        if an.error:
            log.error("model not loaded: %s", an.error)
        yield

    app = FastAPI(title="VeriDoc AI (research prototype)", version="0.5.0", lifespan=lifespan,
                  description="Tampering-localisation heatmaps for document images. Synthetic-data "
                              "prototype; no authenticity verdicts.")
    app.state.analyzer = analyzer or Analyzer(settings)
    app.add_middleware(BodySizeLimit, max_body=up["max_bytes"] + up["multipart_overhead"])

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        an = app.state.analyzer
        body = {"status": "ok" if an.ready else "unavailable", "model_loaded": an.ready,
                "settings": an.settings_summary() if an.ready else None,
                "error": an.error}
        return JSONResponse(body, status_code=200 if an.ready else 503)

    @app.post("/v1/analyze")
    def analyze(file: UploadFile = File(..., description="JPEG or PNG document image, at most 10 MB")):
        an = app.state.analyzer
        rid = uuid.uuid4().hex[:12]
        if not an.ready:
            return JSONResponse(_error("model_unavailable", an.error or "model not loaded"), status_code=503)
        t0 = time.perf_counter()
        try:
            data = file.file.read(up["max_bytes"] + 1)
            img, info = decode_upload(data, up)
        except UploadError as e:
            log.info("analyze rid=%s rejected status=%d code=%s", rid, e.status, e.code)
            return JSONResponse(_error(e.code, e.message), status_code=e.status)
        finally:
            file.file.close()
        del data
        t1 = time.perf_counter()
        prob = an(img)
        t2 = time.perf_counter()
        res = analyze_page(img, prob, an.frozen, outcfg["max_regions"], outcfg["display_max_side"],
                           outcfg["overlay_alpha"])
        body = {
            "request_id": rid,
            "image": {k: info[k] for k in ("format", "width", "height")},
            "risk_score": {"value": round(res["risk_score"], 4), "label": "experimental", "note": RISK_SCORE_NOTE},
            "regions": res["regions"], "n_regions": res["n_regions"],
            "regions_truncated": res["n_regions"] > len(res["regions"]),
            "regions_note": "Connected regions of pixels at or above the frozen pixel threshold; boxes are "
                            "[x0, y0, x1, y1] in original image pixels (x1, y1 exclusive).",
            "predicted_area_fraction": round(res["predicted_area_fraction"], 6),
            "heatmap_png_base64": _b64(res["heatmap"], "PNG"),
            "overlay_jpeg_base64": _b64(res["overlay"], "JPEG", outcfg["overlay_jpeg_quality"]),
            "display_size": [int(res["overlay"].shape[1]), int(res["overlay"].shape[0])],
            "settings": an.settings_summary(),
            "limitations": LIMITATIONS,
        }
        t3 = time.perf_counter()
        body["timing_ms"] = {"decode": round((t1 - t0) * 1e3, 1), "inference": round((t2 - t1) * 1e3, 1),
                             "postprocess": round((t3 - t2) * 1e3, 1)}
        log.info("analyze rid=%s ok bytes=%d size=%dx%d regions=%d ms=%.0f", rid, info["bytes"],
                 info["width"], info["height"], res["n_regions"], (t3 - t0) * 1e3)
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    return app


def __getattr__(name: str):
    """`uvicorn serving.app:app` builds the app on first access (not at import time, so tests
    can import this module without a checkpoint)."""
    if name == "app":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
