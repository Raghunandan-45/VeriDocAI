"""Diagnostic (NOT a selection step): why is ELA weak on this dataset?

Regenerates a few TRAIN sources in memory (same seeds and edits) under controlled
post-processing variants and measures ELA with the frozen ELA parameters:
  standard           : the dataset pipeline (resize 0.8-1.2, blur p=0.3, final JPEG 60-95)
  no_resize_no_blur  : resize scale fixed to 1.0, no blur (stage-1 and final JPEG grids align)
  single_jpeg_no_resize: as above but without the stage-1 JPEG (compression.double = false)
Uses train sources only, never val/test. Pixel F1 here uses an ORACLE threshold chosen on the
same pages, so it is an upper bound for comparison between variants, not a result.

    .venv/bin/python -m research.ela_diagnostic --n-sources 40
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from research.config import load_config, resolve, with_overrides
from research.gen.build_dataset import generate_source, plan
from docforge.ela import ela_map
from docforge.scoring import f1_curve, pixel_hists, roc_auc, top_k_mean

VARIANTS = {
    "standard": {},
    "no_resize_no_blur": {"resize.scale": [1.0, 1.0], "blur.prob": 0.0},
    "single_jpeg_no_resize": {"resize.scale": [1.0, 1.0], "blur.prob": 0.0, "compression.double": False},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sources", type=int, default=40)
    ap.add_argument("--out", default="results/ela_diagnostic.json")
    a = ap.parse_args()
    data_cfg = load_config("configs/data.yaml")
    ela_cfg = load_config("configs/eval.yaml")["ela"]
    n_bins = 1000
    out = {"note": __doc__.strip().splitlines()[0], "split_used": "train", "ela": ela_cfg, "variants": {}}
    for name, ov in VARIANTS.items():
        cfg = with_overrides(data_cfg, ov)
        sources, members = plan(cfg)
        train = [s for s in sources if s["split"] == "train"]
        pick = train[:: max(1, len(train) // a.n_sources)][: a.n_sources]
        labels, smax, stop, pos_all, neg_all = [], [], [], [], []
        for src in pick:
            for rec in generate_source(src, members["train"], cfg, resolve("."), write=False):
                m = ela_map(rec["_image"], **ela_cfg)
                p, n = pixel_hists(m, rec["_mask"], n_bins)
                pos_all.append(p)
                neg_all.append(n)
                labels.append(rec["image_label"])
                smax.append(float(m.max()))
                stop.append(top_k_mean(m, 500))
        curve = f1_curve(np.sum(pos_all, 0), np.sum(neg_all, 0))
        out["variants"][name] = {
            "overrides": ov, "n_pages": len(labels), "n_tampered": int(sum(labels)),
            "image_auc_max": roc_auc(labels, smax), "image_auc_top_k_mean": roc_auc(labels, stop),
            "oracle_pooled_pixel_f1": float(curve.max()), "oracle_threshold": int(curve.argmax()) / n_bins}
        print(name, {k: v for k, v in out["variants"][name].items() if k != "overrides"})
    resolve(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
