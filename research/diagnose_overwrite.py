"""Diagnose why the U-Net misses text overwrites (no GPU, no test split).

    .venv/bin/python -m research.diagnose_overwrite crops            # replay training crops
    .venv/bin/python -m research.diagnose_overwrite preds --checkpoint artifacts/best.pt
    .venv/bin/python -m research.diagnose_overwrite context          # from existing results files

`crops`: replays exactly the page visits of training (VisitDataset item i of epoch e uses
rng([seed, e, i]); same train manifest, same crop config) for all epochs, and counts per
manipulation type: positive crops, positive (mask) pixels, and their share of all positive
pixels seen. Also reports full-page mask areas per type. -> results/diag_overwrite_crops.json

`preds`: runs the checkpoint (CPU, frozen stride) on every VAL text-overwrite page and on a
sample of TRAIN text-overwrite pages, and records where the scores are: inside the mask
(max, mean, fraction above the frozen pixel threshold and above 0.5) versus outside it
(max), whether the page maximum falls inside the mask, and the share of the top-500 pixels
that are inside the mask. Train pages tell underfitting (also missed on train) apart from a
generalisation failure (found on train, missed on val).
-> results/diag_overwrite_preds.json and results/diag_overwrite_preds.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from research.config import load_config, resolve
from research.gen.dataset_hash import read_manifest

TOP_K = 500


# --- crops ------------------------------------------------------------------------------------
def _replay_chunk(args):
    """Replay visits [i0, i1) of one epoch; return per-crop rows (no pixels kept)."""
    records, data_root, crops_cfg, seed, epoch, i0, i1, k = args
    from research.sampler import CropSampler
    s = CropSampler(records, Path(data_root), crops_cfg, cache_pages=2)
    by_id = {r["sample_id"]: r for r in records}
    rows = []
    for i in range(i0, i1):
        for c in s.sample_visit(np.random.default_rng([seed, epoch, i]), k):
            r = by_id[c.sample_id]
            rows.append((r["manipulation_type"], r.get("difficulty"), r.get("splice_size_class"),
                         c.mode, int(c.mask.sum())))
    return rows


def crops(a):
    tcfg = load_config(a.train_config)
    crops_cfg = load_config(tcfg["data_config"])["crops"]
    tr = tcfg["train"]
    data_root = resolve(a.data_root)
    records = read_manifest(data_root / "manifests" / "train.jsonl")
    n, k, seed = tr["visits_per_epoch"], tr["crops_per_visit"], tcfg["seed"]
    epochs = range(tr["epochs"] if a.epochs is None else a.epochs)
    step = 100
    jobs = [(records, str(data_root), crops_cfg, seed, e, i, min(i + step, n), k)
            for e in epochs for i in range(0, n, step)]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for part in ex.map(_replay_chunk, jobs):
            rows += part

    def group_key(row, key):
        mt, diff, size, _, _ = row
        return {"manipulation_type": mt, "difficulty": f"{mt}/{diff}" if diff else mt,
                "splice_size_class": f"{mt}/{size}" if size else mt}[key]

    total_pos_px = sum(r[4] for r in rows)
    total_pos = sum(1 for r in rows if r[3] == "positive")
    out = {"evidence_class": "replay of the training crop stream (train split only)",
           "epochs": len(epochs), "visits_per_epoch": n, "crops_per_visit": k, "seed": seed,
           "n_crops": len(rows), "n_positive_crops": total_pos, "positive_pixels_total": total_pos_px,
           "crops_by_mode": {m: sum(1 for r in rows if r[3] == m) for m in sorted({r[3] for r in rows})},
           "groups": {}}
    for key in ("manipulation_type", "difficulty", "splice_size_class"):
        g = defaultdict(list)
        for r in rows:
            if r[3] == "positive":
                g[group_key(r, key)].append(r[4])
        out["groups"][key] = {name: {
            "positive_crops": len(v),
            "share_of_positive_crops": round(len(v) / total_pos, 4),
            "positive_pixels": int(sum(v)),
            "share_of_positive_pixels": round(sum(v) / total_pos_px, 4),
            "mean_positive_pixels_per_positive_crop": round(float(np.mean(v)), 1),
            "median_positive_pixels_per_positive_crop": float(np.median(v)),
            "mean_positive_fraction_of_crop": round(float(np.mean(v)) / crops_cfg["size"] ** 2, 4),
        } for name, v in sorted(g.items())}
    area = defaultdict(list)
    for r in records:
        if r["image_label"] == 1:
            area[r["manipulation_type"]].append(r["mask_positive_pixels"])
            if r.get("difficulty"):
                area[f"{r['manipulation_type']}/{r['difficulty']}"].append(r["mask_positive_pixels"])
    out["train_page_mask_pixels"] = {t: {"n": len(v), "median": float(np.median(v)), "mean": round(float(np.mean(v)), 1),
                                         "min": int(min(v)), "max": int(max(v))} for t, v in sorted(area.items())}
    path = resolve("results/diag_overwrite_crops.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", path)


# --- predictions ------------------------------------------------------------------------------
def page_stats(prob: np.ndarray, gt: np.ndarray, thr: float) -> dict:
    """Where the scores are, relative to the ground-truth mask (gt: bool)."""
    inside, outside = prob[gt], prob[~gt]
    flat = prob.ravel()
    top = np.argpartition(flat, -TOP_K)[-TOP_K:]
    ys, xs = np.nonzero(gt)
    my, mx = np.unravel_index(int(np.argmax(prob)), prob.shape)
    dy = max(ys.min() - my, 0, my - ys.max())
    dx = max(xs.min() - mx, 0, mx - xs.max())
    return {"mask_pixels": int(gt.sum()),
            "in_max": float(inside.max()), "in_mean": float(inside.mean()), "in_p90": float(np.quantile(inside, 0.9)),
            "in_frac_above_thr": float((inside >= thr).mean()), "in_frac_above_0.5": float((inside >= 0.5).mean()),
            "out_max": float(outside.max()), "out_frac_above_thr": float((outside >= thr).mean()),
            "page_max_inside_mask": bool(gt[my, mx]), "page_max_dist_to_mask_bbox_px": float(np.hypot(dx, dy)),
            "top500_share_inside_mask": float(gt.ravel()[top].mean())}


def classify(s: dict, thr: float) -> str:
    """Rough category per page (thresholds: frozen pixel threshold and 0.5)."""
    if s["in_frac_above_thr"] > 0:
        return "detected (some mask pixels >= frozen thr)"
    if s["in_max"] >= 0.5:
        return "weak response inside mask (0.5 <= in_max < thr)"
    if s["out_max"] >= thr:
        return "missed, high scores elsewhere (misplaced)"
    return "missed, low scores everywhere"


def preds(a):
    import torch
    from docforge.model import load_checkpoint, make_predict_fn, predict_page
    from docforge.preprocess import load_mask, load_rgb
    from research.evaluate import sha256_path

    ecfg = load_config(a.eval_config)
    frozen = json.loads(resolve(a.frozen_config).read_text())
    thr = float(frozen["pixel_threshold"])
    ckpt_sha = sha256_path(resolve(a.checkpoint))
    if ckpt_sha != frozen["checkpoint_sha256"]:
        raise SystemExit("checkpoint does not match the frozen config")
    torch.set_num_threads(a.threads)
    model, meta = load_checkpoint(resolve(a.checkpoint), "cpu")
    fn = make_predict_fn(model, "cpu", False)
    t = ecfg["tiling"]
    data_root = resolve(a.data_root)
    val = [r for r in read_manifest(data_root / "manifests" / "val.jsonl") if r["manipulation_type"] == "text_overwrite"]
    tr = [r for r in read_manifest(data_root / "manifests" / "train.jsonl") if r["manipulation_type"] == "text_overwrite"]
    rng = np.random.default_rng([a.seed, 99])
    pick = []
    for d in ("easy", "hard"):
        pool = sorted([r for r in tr if r["difficulty"] == d], key=lambda r: r["sample_id"])
        pick += [pool[i] for i in sorted(rng.choice(len(pool), a.train_pages // 2, replace=False))]
    rows = []
    for split, recs in (("val", val), ("train", pick)):
        for j, r in enumerate(recs):
            img = load_rgb(data_root / r["image_path"])
            gt = load_mask(data_root / r["mask_path"])
            prob = predict_page(img, fn, ecfg["analysis"]["max_side"], t["tile"], frozen["stride"],
                                t["batch_size"], t["window_floor"])
            s = page_stats(prob, gt, thr)
            e = r["edit"]
            row = {"split": split, "sample_id": r["sample_id"], "difficulty": r["difficulty"],
                   "field_kind": e.get("field_kind"), "imperfections": "+".join(e.get("imperfections") or []),
                   "font_change": e.get("orig_font_family") != e.get("edit_font_family"),
                   "size_change_px": (e.get("edit_font_size") or 0) - (e.get("orig_font_size") or 0), **s}
            row["category"] = classify(s, thr)
            rows.append(row)
            print(f"[{split} {j + 1}/{len(recs)}] {r['sample_id']} {r['difficulty']:4s} in_max {s['in_max']:.3f} "
                  f"out_max {s['out_max']:.3f} -> {row['category']}", flush=True)

    def summary(sel):
        if not sel:
            return None
        cats = defaultdict(int)
        for r in sel:
            cats[r["category"]] += 1
        num = ("in_max", "in_mean", "in_frac_above_thr", "out_max", "top500_share_inside_mask")
        return {"n": len(sel), "categories": dict(sorted(cats.items())),
                "page_max_inside_mask_frac": round(float(np.mean([r["page_max_inside_mask"] for r in sel])), 4),
                **{f"median_{k}": round(float(np.median([r[k] for r in sel])), 4) for k in num}}

    out = {"evidence_class": "SYNTHETIC; diagnostic only (val text-overwrite pages + a train sample); no test data",
           "checkpoint_sha256": ckpt_sha, "checkpoint_epoch": meta.get("epoch"), "pixel_threshold": thr,
           "stride": frozen["stride"], "train_sample_seed": a.seed, "summary": {}}
    for split in ("val", "train"):
        for d in (None, "easy", "hard"):
            sel = [r for r in rows if r["split"] == split and (d is None or r["difficulty"] == d)]
            out["summary"][f"{split}/{d or 'all'}"] = summary(sel)
    path = resolve("results/diag_overwrite_preds.json")
    path.write_text(json.dumps(out, indent=2))
    with open(resolve("results/diag_overwrite_preds.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(out["summary"], indent=2))
    print("wrote", path, "and results/diag_overwrite_preds.csv")


# --- context ----------------------------------------------------------------------------------
def context(a):
    """How much the per-epoch selection metric can see of text overwrites, and how the image
    scores of each page type compare (from existing results files only)."""
    import pandas as pd
    import yaml
    data_root = resolve(a.data_root)
    val = {r["sample_id"]: r for r in read_manifest(data_root / "manifests" / "val.jsonl")}
    ids = yaml.safe_load(resolve(a.train_config_used).read_text())["val_subset_ids"]

    def mix(sample_ids):
        n = defaultdict(int)
        px = defaultdict(int)
        for i in sample_ids:
            n[val[i]["manipulation_type"]] += 1
            px[val[i]["manipulation_type"]] += val[i]["mask_positive_pixels"]
        tot = px["splice"] + px["text_overwrite"]
        return {"pages": dict(sorted(n.items())), "mask_pixels": {k: px[k] for k in ("splice", "text_overwrite")},
                "text_overwrite_share_of_mask_pixels": round(px["text_overwrite"] / tot, 4)}

    df = pd.read_csv(resolve(a.per_image))
    df["type"] = df["sample_id"].map(lambda s: val[s]["manipulation_type"])
    med = df.groupby("type")[["score_max", "score_top_k_mean"]].median().round(4)
    m = pd.read_csv(resolve(a.metrics))
    out = {"evidence_class": "SYNTHETIC validation; derived from existing results files",
           "val_subset_used_for_checkpoint_selection": mix(ids), "full_val": mix(list(val)),
           "median_image_scores_by_type": med.to_dict(orient="index"),
           "metrics_csv": m[["epoch", "train_loss", "val_subset_best_f1", "lr_end"]].to_dict(orient="records")}
    path = resolve("results/diag_overwrite_context.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    x = sub.add_parser("context")
    x.add_argument("--data-root", default="data")
    x.add_argument("--train-config-used", default="results/kaggle_run/config_used.yaml")
    x.add_argument("--per-image", default="results/val_model_per_image.csv")
    x.add_argument("--metrics", default="results/kaggle_run/metrics.csv")
    c = sub.add_parser("crops")
    c.add_argument("--train-config", default="configs/train.yaml")
    c.add_argument("--data-root", default="data")
    c.add_argument("--epochs", type=int, default=None, help="default: all training epochs")
    c.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("preds")
    p.add_argument("--checkpoint", default="artifacts/best.pt")
    p.add_argument("--frozen-config", default="results/frozen_config.json")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--data-root", default="data")
    p.add_argument("--train-pages", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    {"crops": crops, "preds": preds, "context": context}[a.cmd](a)


if __name__ == "__main__":
    main()
