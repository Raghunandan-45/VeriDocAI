"""Full-page evaluation harness with a test-set guard.

    # validation: select pixel threshold, image-score rule and operating point, then freeze
    .venv/bin/python -m research.evaluate --method ela --split val
    # test (Phase 4): ONLY with a frozen config produced from validation
    .venv/bin/python -m research.evaluate --method ela --split test --frozen-config results/frozen_ela.json
    # trained model (CPU is fine): validation selection -> results/frozen_config.json
    .venv/bin/python -m research.evaluate --method model --checkpoint artifacts/best.pt --split val

Protocol:
- On "val": choose the global pixel threshold (max pooled pixel F1 over all bin edges), then
  the image-score rule (max / top-k mean / fraction above pixel threshold) by ROC-AUC, then the
  image threshold reaching the target recall on tampered pages. These are written to a frozen
  JSON. Validation metrics are computed with the thresholds selected on the same data, so
  they are optimistic and labelled as such.
- On any split whose name starts with "test": selection is refused. A --frozen-config whose
  `selected_on` is "val" (and whose method/ELA parameters match) is required, and only the
  frozen rule's numbers are reported.
All results are SYNTHETIC evaluation, not evidence of real-world performance.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp  # spawn: forking after torch/OpenMP init can deadlock
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from research.config import load_config, resolve
from docforge.ela import ela_map
from docforge.preprocess import load_mask, load_rgb
from docforge.scoring import (SCORE_RULES, bootstrap_ci, frac_above, pixel_hists, prf_iou, recall_fpr_at,
                              roc_auc, select_operating_point, select_pixel_threshold, threshold_bin, top_k_mean)

GROUP_KEYS = ("manipulation_type", "difficulty", "splice_size_class", "document_type")


class TestSetGuardError(RuntimeError):
    """Raised when someone tries to select thresholds/rules on a test split."""


def is_test_split(split: str) -> bool:
    return split.startswith("test")


def check_guard(split: str, frozen: dict | None, method: str, cfg: dict, model_info: dict | None = None) -> None:
    """Refuse test-split evaluation without a validation-frozen config, and refuse frozen
    configs that were not selected on validation or do not match the method/parameters."""
    if is_test_split(split) and frozen is None:
        raise TestSetGuardError(
            f"refusing to evaluate split '{split}' without --frozen-config: thresholds and score rules "
            "may only be selected on validation (run --split val first)")
    if frozen is not None:
        if frozen.get("selected_on") != "val":
            raise TestSetGuardError(f"frozen config was selected on {frozen.get('selected_on')!r}, not 'val'")
        if frozen.get("method") != method:
            raise TestSetGuardError(f"frozen config is for method {frozen.get('method')!r}, not {method!r}")
        if method == "ela" and frozen.get("ela") != cfg["ela"]:
            raise TestSetGuardError("ELA parameters in configs/eval.yaml differ from the frozen config")
        if method == "model" and model_info is not None:
            for key in ("checkpoint_sha256", "stride", "max_side"):
                if frozen.get(key) != model_info[key]:
                    raise TestSetGuardError(f"frozen {key}={frozen.get(key)!r} but this run uses {model_info[key]!r}")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- per-image statistics ------------------------------------------------------------------
def ela_score_map(img: np.ndarray, params: dict) -> np.ndarray:
    return ela_map(img, **params)


class ModelScoreMap:
    """Picklable full-page predictor: loads the checkpoint lazily (CPU unless CUDA exists)."""

    def __init__(self, checkpoint: str, cfg: dict, stride: int):
        self.checkpoint, self.cfg, self.stride = checkpoint, cfg, stride
        self.fn = None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.fn is None:
            import torch
            from docforge.model import load_checkpoint, make_predict_fn
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = load_checkpoint(self.checkpoint, device)
            self.fn = make_predict_fn(model, device, amp=device == "cuda")
        from docforge.model import predict_page
        t = self.cfg["tiling"]
        return predict_page(img, self.fn, self.cfg["analysis"]["max_side"], t["tile"], self.stride,
                            t["batch_size"], t["window_floor"])


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_info(checkpoint: str, cfg: dict, stride: int | None) -> dict:
    """Identity of the model run, recorded in (and checked against) the frozen config."""
    import torch
    ck = torch.load(resolve(checkpoint), map_location="cpu", weights_only=True)
    return {"checkpoint": checkpoint, "checkpoint_sha256": sha256_path(resolve(checkpoint)),
            "train_config_sha256": sha256_bytes(json.dumps(ck.get("train_config"), sort_keys=True).encode()),
            "checkpoint_epoch": ck.get("epoch"), "stride": stride or cfg["tiling"]["stride"],
            "tile": cfg["tiling"]["tile"], "max_side": cfg["analysis"]["max_side"]}


def image_stats(rec: dict, data_root: Path, map_fn, n_bins: int, top_k: int) -> dict:
    """Score map for one page reduced to histograms + max + top-k mean (small to keep)."""
    img = load_rgb(data_root / rec["image_path"])
    gt = load_mask(data_root / rec["mask_path"])
    prob = map_fn(img)
    if prob.shape != gt.shape:
        raise ValueError(f"{rec['sample_id']}: score map {prob.shape} != mask {gt.shape}")
    pos, neg = pixel_hists(prob, gt, n_bins)
    return {"sample_id": rec["sample_id"], "label": int(rec["image_label"]), "pos": pos, "neg": neg,
            "max": float(prob.max()), "top_k_mean": top_k_mean(prob, top_k)}


def _stats_job(args):
    return image_stats(*args)


def collect(records, data_root: Path, map_fn, cfg: dict, workers: int) -> list[dict]:
    jobs = [(r, data_root, map_fn, cfg["thresholds"]["n_bins"], cfg["image_score"]["top_k"]) for r in records]
    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            return list(ex.map(_stats_job, jobs, chunksize=4))
    return [_stats_job(j) for j in jobs]


def image_scores(stats: list[dict], rule: str, k_bin: int) -> np.ndarray:
    if rule == "max":
        return np.array([s["max"] for s in stats])
    if rule == "top_k_mean":
        return np.array([s["top_k_mean"] for s in stats])
    if rule == "frac_above":
        return np.array([frac_above(s["pos"], s["neg"], k_bin) for s in stats])
    raise ValueError(rule)


# --- selection (validation only) -------------------------------------------------------------
def select_on_validation(stats: list[dict], cfg: dict) -> dict:
    n_bins = cfg["thresholds"]["n_bins"]
    sel = select_pixel_threshold([s["pos"] for s in stats], [s["neg"] for s in stats], n_bins)
    k = threshold_bin(sel["pixel_threshold"], n_bins)
    labels = np.array([s["label"] for s in stats])
    aucs = {r: roc_auc(labels, image_scores(stats, r, k)) for r in cfg["image_score"]["rules"]}
    valid = {r: a for r, a in aucs.items() if a is not None}
    rule = max(valid, key=lambda r: (valid[r], -cfg["image_score"]["rules"].index(r)))
    o = cfg["operating_point"]
    op = select_operating_point(image_scores(stats, rule, k), labels, o["target_recall"],
                                o["min_tampered"], o["min_authentic"], o["min_unique_scores"])
    return {**sel, "val_auc_by_rule": aucs, "score_rule": rule, "operating_point": op,
            "image_threshold": op["image_threshold"], "target_recall": o["target_recall"]}


# --- metrics with a fixed (frozen or just-selected) configuration -----------------------------
def summarize(idx: np.ndarray, ctx: dict, thr: float | None, auc_idx: np.ndarray, bcfg: dict) -> dict:
    """Metrics for the images `idx`; image AUC uses `auc_idx` (group's tampered + all authentic)."""
    tp, fp, fn, f1, iou, y, s = (ctx[k] for k in ("tp", "fp", "fn", "f1", "iou", "y", "s"))
    nb, seed, lvl = bcfg["n"], bcfg["seed"], bcfg["level"]
    tam = idx[y[idx] == 1]
    pooled = prf_iou(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())
    out = {"n_images": int(len(idx)), "n_tampered": int(len(tam)), "n_authentic": int(len(idx) - len(tam)),
           "pixel_pooled": pooled,
           "pixel_pooled_f1_ci": bootstrap_ci(len(idx), lambda b: prf_iou(tp[idx[b]].sum(), fp[idx[b]].sum(),
                                                                          fn[idx[b]].sum())["f1"], nb, seed, lvl)}
    if len(tam):
        out["pixel_per_image_mean_over_tampered"] = {"f1": float(f1[tam].mean()), "iou": float(iou[tam].mean())}
        out["pixel_per_image_f1_ci"] = bootstrap_ci(len(tam), lambda b: float(f1[tam[b]].mean()), nb, seed, lvl)
    out["image_auc"] = roc_auc(y[auc_idx], s[auc_idx])
    out["image_auc_ci"] = bootstrap_ci(len(auc_idx), lambda b: roc_auc(y[auc_idx[b]], s[auc_idx[b]]), nb, seed, lvl)
    if thr is not None:
        rf = recall_fpr_at(s[auc_idx], y[auc_idx], thr)
        out["at_frozen_image_threshold"] = {
            **rf,
            "recall_ci": bootstrap_ci(len(tam), lambda b: float((s[tam[b]] >= thr).mean()), nb, seed, lvl)
            if len(tam) else None}
    return out


def evaluate(stats: list[dict], records: list[dict], frozen: dict, cfg: dict) -> dict:
    n_bins = frozen["n_bins"]
    k = threshold_bin(frozen["pixel_threshold"], n_bins)
    tp, fp, fn = (np.array(v) for v in zip(*[(s["pos"][k:].sum(), s["neg"][k:].sum(), s["pos"][:k].sum())
                                            for s in stats]))
    m = prf_iou(tp, fp, fn)
    y = np.array([s["label"] for s in stats])
    s_img = image_scores(stats, frozen["score_rule"], k)
    ctx = {"tp": tp, "fp": fp, "fn": fn, "f1": m["f1"], "iou": m["iou"], "y": y, "s": s_img}
    thr = frozen.get("image_threshold")
    allidx = np.arange(len(stats))
    auth = allidx[y == 0]
    b = cfg["bootstrap"]
    out = {"overall": summarize(allidx, ctx, thr, allidx, b)}
    if thr is not None:
        out["overall"]["at_frozen_image_threshold"]["fpr_ci"] = bootstrap_ci(
            len(auth), lambda bb: float((s_img[auth[bb]] >= thr).mean()), b["n"], b["seed"], b["level"])
    area = np.array([(st["pos"][k:].sum() + st["neg"][k:].sum()) / (st["pos"].sum() + st["neg"].sum())
                     for st in stats])
    out["authentic_pages"] = {"n": int(len(auth)),
                              "frac_with_any_predicted_pixel": float((area[auth] > 0).mean()) if len(auth) else None,
                              "mean_predicted_area_fraction": float(area[auth].mean()) if len(auth) else None}
    by = {}
    for key in GROUP_KEYS:
        values = sorted({r.get(key) for r in records if r["image_label"] == 1 and r.get(key) is not None})
        for v in values:
            gidx = allidx[[r["image_label"] == 1 and r.get(key) == v for r in records]]
            by[f"{key}={v}"] = summarize(gidx, ctx, thr, np.concatenate([gidx, auth]), b)
    out["breakdown_tampered_groups"] = by
    out["per_image"] = [{"sample_id": st["sample_id"], "label": int(y[i]), "score": float(s_img[i]),
                         "score_max": st["max"], "score_top_k_mean": st["top_k_mean"],
                         "score_frac_above": float(image_scores([st], "frac_above", k)[0]),
                         "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                         "f1": float(m["f1"][i]), "iou": float(m["iou"][i]), "pred_area_fraction": float(area[i])}
                        for i, st in enumerate(stats)]
    return out


def _round(o, nd=5):
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, list):
        return [_round(v, nd) for v in o]
    return o


def run(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["ela", "model"], required=True)
    ap.add_argument("--checkpoint", default=None, help="model only, e.g. artifacts/best.pt")
    ap.add_argument("--stride", type=int, default=None, help="model only; default tiling.stride")
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--frozen-config", default=None, help="JSON frozen on validation (required for test splits)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=None, help="default results/<split>_<method>.json")
    ap.add_argument("--frozen-out", default=None,
                    help="val only; default results/frozen_ela.json or results/frozen_config.json (model)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N records only")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    frozen = json.loads(resolve(a.frozen_config).read_text()) if a.frozen_config else None
    check_guard(a.split, frozen, a.method, cfg)  # BEFORE any data is touched
    minfo = None
    if a.method == "model":
        if not a.checkpoint:
            ap.error("--method model needs --checkpoint")
        minfo = model_info(a.checkpoint, cfg, a.stride)
        check_guard(a.split, frozen, a.method, cfg, minfo)

    data_root = resolve(a.data_root or cfg["data_root"])
    records = read_manifest(data_root / "manifests" / f"{a.split}.jsonl")
    records = sorted(records, key=lambda r: r["sample_id"])[: a.limit]
    if a.method == "ela":
        map_fn = partial(ela_score_map, params=cfg["ela"])
        workers = cfg["workers"] if a.workers is None else a.workers
    else:  # one process; torch uses its own intra-op threads
        map_fn = ModelScoreMap(str(resolve(a.checkpoint)), cfg, minfo["stride"])
        workers = 0
    stats = collect(records, data_root, map_fn, cfg, workers)

    hash_file = resolve(load_config(cfg["data_config"])["paths"]["hash_file"])
    provenance = {"dataset_sha256": dict(l.split("=", 1) for l in hash_file.read_text().split())["sha256"],
                  "eval_config_sha256": sha256_bytes(resolve(a.config).read_bytes()),
                  "limit": a.limit, **({"ela": cfg["ela"]} if a.method == "ela" else minfo)}
    result = {"method": a.method, "split": a.split, **provenance}
    if frozen is None:  # validation: select, freeze, then report (optimistic) val metrics
        sel = select_on_validation(stats, cfg)
        frozen = {"method": a.method, "selected_on": a.split, "pixel_threshold": sel["pixel_threshold"],
                  "n_bins": sel["n_bins"], "score_rule": sel["score_rule"],
                  "image_threshold": sel["image_threshold"], "target_recall": sel["target_recall"],
                  "operating_point_reliable": sel["operating_point"]["reliable"],
                  "operating_point_reasons": sel["operating_point"]["reasons"],
                  "top_k": cfg["image_score"]["top_k"], **provenance}
        result["evidence_class"] = ("SYNTHETIC validation; thresholds/rule selected on this same data, "
                                    "so these numbers are optimistic")
        result["selection"] = sel
        fpath = resolve(a.frozen_out or ("results/frozen_ela.json" if a.method == "ela" else "results/frozen_config.json"))
        if a.limit is None or a.frozen_out:
            fpath.write_text(json.dumps(_round(frozen, 8), indent=2))
            result["frozen_config_path"] = os.path.relpath(fpath, resolve("."))
    else:
        result["evidence_class"] = "SYNTHETIC held-out evaluation with a validation-frozen config"
        result["frozen_config"] = frozen
    metrics = evaluate(stats, records, frozen, cfg)
    per_image = metrics.pop("per_image")
    result["metrics"] = metrics
    out = resolve(a.out or f"results/{a.split}_{a.method}.json")
    out.write_text(json.dumps(_round(result), indent=2))
    meta = {r["sample_id"]: r for r in records}
    with open(out.with_name(out.stem + "_per_image.csv"), "w", newline="") as f:
        cols = ["sample_id", "document_type", "manipulation_type", "difficulty", "splice_size_class"]
        w = csv.DictWriter(f, fieldnames=cols + [k for k in per_image[0] if k != "sample_id"])
        w.writeheader()
        for row in per_image:
            r = meta[row["sample_id"]]
            w.writerow({**{c: r.get(c) for c in cols}, **_round(row)})
    print(f"wrote {out} (+ per-image CSV)")
    return result


if __name__ == "__main__":
    run()
