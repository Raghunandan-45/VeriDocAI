"""Verify training artifacts copied from Kaggle into artifacts/.

    .venv/bin/python -m research.verify_artifacts --dir artifacts

1. Every file listed in SHA256SUMS.txt (written by research/train.py on the kernel) exists
   and has the same sha256 locally; required files are present.
2. best.pt loads on CPU without network access (encoder_weights=None) and produces a
   full-page probability map of the right shape and range on one validation page.
Writes results/artifacts_verification.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from research.config import load_config, resolve
from research.evaluate import read_manifest, sha256_path
from docforge.model import load_checkpoint, make_predict_fn, predict_page
from docforge.preprocess import load_rgb

REQUIRED = ["best.pt", "last.pt", "config_used.yaml", "metrics.csv", "SHA256SUMS.txt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="results/artifacts_verification.json")
    a = ap.parse_args()
    d = resolve(a.dir)
    report = {"dir": a.dir, "missing": [f for f in REQUIRED if not (d / f).exists()], "files": {}}
    if (d / "SHA256SUMS.txt").exists():
        for line in (d / "SHA256SUMS.txt").read_text().splitlines():
            expected, name = line.split(maxsplit=1)
            p = d / name
            got = sha256_path(p) if p.exists() else None
            report["files"][name] = {"expected": expected, "local": got, "ok": got == expected,
                                     "bytes": p.stat().st_size if p.exists() else None}
    report["all_hashes_ok"] = bool(report["files"]) and all(f["ok"] for f in report["files"].values())

    model, meta = load_checkpoint(d / "best.pt", "cpu")
    report["checkpoint"] = {k: meta.get(k) for k in ("epoch", "val_metric", "val_value", "val_subset_threshold", "arch")}
    rec = sorted(read_manifest(resolve(a.data_root) / "manifests" / "val.jsonl"), key=lambda r: r["sample_id"])[0]
    img = load_rgb(resolve(a.data_root) / rec["image_path"])
    ev = load_config("configs/eval.yaml")
    t0 = time.time()
    prob = predict_page(img, make_predict_fn(model, "cpu"), ev["analysis"]["max_side"], ev["tiling"]["tile"],
                        ev["tiling"]["stride"], ev["tiling"]["batch_size"], ev["tiling"]["window_floor"])
    report["cpu_forward"] = {"sample_id": rec["sample_id"], "image_hw": list(img.shape[:2]),
                             "prob_hw": list(prob.shape), "prob_min": float(prob.min()), "prob_max": float(prob.max()),
                             "finite": bool(np.isfinite(prob).all()), "stride": ev["tiling"]["stride"],
                             "seconds": round(time.time() - t0, 2)}
    report["ok"] = (not report["missing"] and report["all_hashes_ok"] and report["cpu_forward"]["finite"]
                    and report["cpu_forward"]["prob_hw"] == report["cpu_forward"]["image_hw"])
    resolve(a.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("ARTIFACTS OK" if report["ok"] else "ARTIFACTS NOT OK")
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
