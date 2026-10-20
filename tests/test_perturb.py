"""Quick robustness check: perturbations are deterministic, in range, keep masks aligned,
and can only be used together with a validation-frozen config."""
from __future__ import annotations

import json

import numpy as np
import pytest

from research import evaluate as ev
from research.config import load_config
from research.perturb import perturb


@pytest.fixture(scope="module")
def pcfg():
    return load_config("configs/robustness.yaml")


def _page():
    rng = np.random.default_rng(0)
    img = np.full((400, 300, 3), 235, np.uint8)
    img[100:120, 40:260] = rng.integers(0, 80, (20, 220, 3), dtype=np.uint8)  # a "text line"
    mask = np.zeros((400, 300), bool)
    mask[200:240, 60:180] = True
    return img, mask


@pytest.mark.parametrize("name,key", [("jpeg", "jpeg_quality"), ("resize", "resize_scale"), ("blur", "blur_sigma")])
def test_perturbation_deterministic_and_in_range(pcfg, name, key):
    img, mask = _page()
    a = perturb(img, mask, name, pcfg, "invoice_0001_splice_0")
    b = perturb(img, mask, name, pcfg, "invoice_0001_splice_0")
    assert a[2] == b[2] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    lo, hi = pcfg[name][{"jpeg": "quality", "resize": "scale", "blur": "sigma"}[name]]
    assert lo <= a[2][key] <= hi
    assert not np.array_equal(a[0], img)                       # the image really changed
    values = {perturb(img, mask, name, pcfg, f"s{i}")[2][key] for i in range(20)}
    assert len(values) > 1                                     # parameters vary across pages


def test_resize_keeps_mask_aligned_others_keep_mask(pcfg):
    img, mask = _page()
    out, m, p = perturb(img, mask, "resize", pcfg, "x")
    s = p["resize_scale"]
    assert out.shape[:2] == m.shape and m.dtype == bool
    ys, xs = np.nonzero(m)
    assert abs(ys.min() - 200 * s) <= 2 and abs(xs.max() + 1 - 180 * s) <= 2
    for name in ("jpeg", "blur"):
        assert np.array_equal(perturb(img, mask, name, pcfg, "x")[1], mask)


def test_unknown_perturbation_raises(pcfg):
    img, mask = _page()
    with pytest.raises(ValueError):
        perturb(img, mask, "rotate", pcfg, "x")


def test_cli_refuses_perturb_without_frozen_config(monkeypatch):
    monkeypatch.setattr(ev, "read_manifest", lambda p: pytest.fail("data touched"))
    with pytest.raises(ev.TestSetGuardError):
        ev.run(["--method", "ela", "--split", "val", "--perturb", "jpeg"])


def test_perturbed_run_uses_frozen_config_end_to_end(tiny_dataset, tmp_path):
    """ELA on the tiny dataset: freeze on val, then a perturbed val run reports the frozen
    values, records the perturbation and never selects."""
    root, _, _ = tiny_dataset
    fz, out = tmp_path / "frozen.json", tmp_path / "val.json"
    ev.run(["--method", "ela", "--split", "val", "--data-root", str(root), "--out", str(out),
            "--frozen-out", str(fz), "--workers", "0"])
    frozen = json.loads(fz.read_text())
    pout = tmp_path / "val_blur.json"
    res = ev.run(["--method", "ela", "--split", "val", "--data-root", str(root), "--frozen-config", str(fz),
                  "--perturb", "blur", "--out", str(pout), "--workers", "0"])
    assert res["frozen_config"]["pixel_threshold"] == frozen["pixel_threshold"]
    assert "selection" not in res and res["perturbation"]["name"] == "blur"
    assert 1.2 <= res["perturbation"]["param_min"] <= res["perturbation"]["param_max"] <= 1.6
    assert "blur_sigma" in (tmp_path / "val_blur_per_image.csv").read_text().splitlines()[0]
