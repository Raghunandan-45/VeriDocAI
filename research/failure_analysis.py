"""Failure analysis for the final model (Phase 4). Never changes a threshold or the model.

    # where scores land on VALIDATION pages: authentic false positives by page structure, and
    # which part of a text-overwrite mask is detected (new text ink / text box / cover only)
    .venv/bin/python -m research.failure_analysis regions --split val
    # per-imperfection / per-easy-mode results for text overwrites (from per-image CSVs)
    .venv/bin/python -m research.failure_analysis imperfections
    # annotated examples from the frozen TEST evaluation (panels + contact sheet)
    .venv/bin/python -m research.failure_analysis examples

The model, checkpoint and pixel threshold come from results/final_model.json and its frozen
config. Explanations written from these outputs are hypotheses, not established causes.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from research.config import load_config, resolve
from research.evaluate import read_manifest, sha256_path

CATEGORIES = ("solid_fill", "rule_line", "text", "blank_paper")


# --- shared -----------------------------------------------------------------------------------
def final_model() -> tuple[dict, dict]:
    fm = json.loads(resolve("results/final_model.json").read_text())
    frozen = json.loads(resolve(fm["frozen_config_for_phase4"]).read_text())
    if sha256_path(resolve(fm["checkpoint_for_phase4"])) != frozen["checkpoint_sha256"]:
        raise SystemExit("checkpoint does not match the frozen config")
    return fm, frozen


def predictor(fm: dict, frozen: dict, threads: int):
    import torch
    from docforge.model import load_checkpoint, make_predict_fn, predict_page
    torch.set_num_threads(threads)
    model, _ = load_checkpoint(resolve(fm["checkpoint_for_phase4"]), "cpu")
    fn = make_predict_fn(model, "cpu", False)
    ecfg = load_config("configs/eval.yaml")
    t = ecfg["tiling"]
    return lambda img: predict_page(img, fn, ecfg["analysis"]["max_side"], t["tile"], frozen["stride"],
                                    t["batch_size"], t["window_floor"])


def structure_maps(img: np.ndarray, diff_thr: int = 40, solid_k: int = 9, line_len: int = 41, grow: int = 5) -> dict:
    """Boolean maps of page structure (each dilated by `grow` px), by distance from paper colour.
    solid_fill: what survives a solid_k x solid_k opening (bands, filled cells, logos);
    rule_line: long thin horizontal/vertical runs; text: remaining non-paper pixels."""
    paper = np.median(img.reshape(-1, 3), axis=0)
    ink = (np.abs(img.astype(np.int16) - paper).max(axis=2) > diff_thr).astype(np.uint8)
    solid = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((solid_k, solid_k), np.uint8))
    rest = ink & (1 - solid)
    lines = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((1, line_len), np.uint8)) | \
            cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((line_len, 1), np.uint8))
    text = rest & (1 - lines)
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    return {"solid_fill": cv2.dilate(solid, k) > 0, "rule_line": cv2.dilate(lines, k) > 0,
            "text": cv2.dilate(text, k) > 0, "ink": ink > 0}


def categorize(st: dict) -> np.ndarray:
    """Per-pixel category index (priority solid_fill > rule_line > text > blank_paper)."""
    cat = np.full(st["ink"].shape, 3, np.uint8)
    for i, name in reversed(list(enumerate(CATEGORIES[:3]))):
        cat[st[name]] = i
    return cat


# --- regions ----------------------------------------------------------------------------------
def regions(a):
    fm, frozen = final_model()
    thr = float(frozen["pixel_threshold"])
    pred = predictor(fm, frozen, a.threads)
    from docforge.preprocess import load_mask, load_rgb
    data_root = resolve("data")
    recs = sorted(read_manifest(data_root / "manifests" / f"{a.split}.jsonl"), key=lambda r: r["sample_id"])
    auth = [r for r in recs if r["image_label"] == 0]
    over = [r for r in recs if r["manipulation_type"] == "text_overwrite"]
    fp_px, area_px = np.zeros(4), np.zeros(4)
    top_zone_fp, border_fp, total_fp = 0, 0, 0
    pages, comp_cats = [], defaultdict(int)
    for j, r in enumerate(auth):
        img = load_rgb(data_root / r["image_path"])
        prob = pred(img)
        fp = prob >= thr
        cat = categorize(structure_maps(img))
        H, W = fp.shape
        area_px += np.bincount(cat.ravel(), minlength=4)
        fp_px += np.bincount(cat[fp], minlength=4)
        total_fp += int(fp.sum())
        top_zone_fp += int(fp[: int(0.15 * H)].sum())
        b = 32
        border = np.zeros_like(fp)
        border[:b], border[-b:], border[:, :b], border[:, -b:] = True, True, True, True
        border_fp += int((fp & border).sum())
        n, lab, stats_, cents = cv2.connectedComponentsWithStats(fp.astype(np.uint8), connectivity=8)
        comps = []
        for c in range(1, n):
            x, y, w, h, area = (int(v) for v in stats_[c])
            ccat = CATEGORIES[int(np.bincount(cat[lab == c], minlength=4).argmax())]
            comp_cats[ccat] += 1
            comps.append({"bbox_xywh": [x, y, w, h], "area": area, "majority_category": ccat,
                          "max_prob": round(float(prob[lab == c].max()), 4)})
        pages.append({"sample_id": r["sample_id"], "header_style": r["layout"].get("header_style") or "none",
                      "table_style": r["layout"].get("table_style"), "fp_pixels": int(fp.sum()),
                      "n_fp_components": len(comps), "page_max": round(float(prob.max()), 4),
                      "components": sorted(comps, key=lambda c: -c["area"])[:10]})
        print(f"[auth {j + 1}/{len(auth)}] {r['sample_id']} fp {int(fp.sum())} comps {len(comps)}", flush=True)

    over_rows = []
    for j, r in enumerate(over):
        img = load_rgb(data_root / r["image_path"])
        gt = load_mask(data_root / r["mask_path"])
        prob = pred(img)
        e = r["edit"]
        s = r["image_size_wh"][0] / r["render_size_wh"][0]
        box = lambda b: [int(round(v * s)) for v in b]  # noqa: E731
        tb = np.zeros_like(gt)
        x0, y0, x1, y1 = box(e["text_box_render_xyxy"])
        tb[y0:y1, x0:x1] = True
        tb &= gt
        ink = structure_maps(img)["ink"]
        parts = {"new_text_ink": tb & ink, "text_box_background": tb & ~ink, "cover_only": gt & ~tb}
        row = {"sample_id": r["sample_id"], "difficulty": r["difficulty"], "mask_pixels": int(gt.sum()),
               "detected_pixels": int((prob[gt] >= thr).sum())}
        for name, m in parts.items():
            row[f"{name}_pixels"] = int(m.sum())
            row[f"{name}_detected"] = int((prob[m] >= thr).sum())
            row[f"{name}_mean_score"] = round(float(prob[m].mean()), 4) if m.any() else None
        over_rows.append(row)
        print(f"[overwrite {j + 1}/{len(over)}] {r['sample_id']} detected {row['detected_pixels']}/{row['mask_pixels']}",
              flush=True)

    def part_summary(rows):
        out = {}
        for name in ("new_text_ink", "text_box_background", "cover_only"):
            px = sum(r[f"{name}_pixels"] for r in rows)
            det = sum(r[f"{name}_detected"] for r in rows)
            out[name] = {"pixels": px, "share_of_mask_pixels": round(px / max(sum(r["mask_pixels"] for r in rows), 1), 4),
                         "pooled_recall_at_thr": round(det / max(px, 1), 4),
                         "share_of_detected_pixels": round(det / max(sum(r["detected_pixels"] for r in rows), 1), 4),
                         "median_mean_score": float(np.median([r[f"{name}_mean_score"] for r in rows
                                                               if r[f"{name}_mean_score"] is not None]))}
        return out

    out = {"evidence_class": f"SYNTHETIC {a.split} pages; final model with frozen pixel threshold; diagnostic only",
           "split": a.split, "model": fm["chosen"], "checkpoint_sha256": frozen["checkpoint_sha256"],
           "pixel_threshold": thr,
           "structure_method": "non-paper = max channel diff from page median > 40; solid_fill = 9x9 opening; "
                               "rule_line = 41 px horizontal/vertical opening of the rest; text = remaining "
                               "non-paper; each grown by 5 px; priority solid_fill > rule_line > text > blank_paper",
           "authentic": {
               "n_pages": len(auth), "pages_with_fp": sum(p["fp_pixels"] > 0 for p in pages),
               "fp_pixels_total": total_fp,
               "fp_pixel_share_by_category": {c: round(fp_px[i] / max(total_fp, 1), 4) for i, c in enumerate(CATEGORIES)},
               "page_area_share_by_category": {c: round(area_px[i] / area_px.sum(), 4) for i, c in enumerate(CATEGORIES)},
               "fp_rate_lift_vs_area": {c: round((fp_px[i] / max(total_fp, 1)) / (area_px[i] / area_px.sum()), 2)
                                        for i, c in enumerate(CATEGORIES)},
               "fp_components_by_majority_category": dict(comp_cats),
               "fp_pixel_share_top_15pct_of_page": round(top_zone_fp / max(total_fp, 1), 4),
               "fp_pixel_share_within_32px_of_border": round(border_fp / max(total_fp, 1), 4),
               "median_fp_pixels_per_page": float(np.median([p["fp_pixels"] for p in pages])),
               "fp_pixels_by_header_style": {k: int(sum(p["fp_pixels"] for p in pages if p["header_style"] == k))
                                             for k in sorted({p["header_style"] for p in pages})},
               "pages_by_header_style": {k: sum(p["header_style"] == k for p in pages)
                                         for k in sorted({p["header_style"] for p in pages})},
               "pages": pages},
           "text_overwrite_mask_parts": {d: part_summary([r for r in over_rows if d == "all" or r["difficulty"] == d])
                                         for d in ("all", "easy", "hard")},
           "text_overwrite_pages": over_rows}
    path = resolve(f"results/failure_regions_{a.split}.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out["authentic"].items() if k != "pages"}, indent=2))
    print(json.dumps(out["text_overwrite_mask_parts"], indent=2))
    print("wrote", path)


# --- imperfections ----------------------------------------------------------------------------
def imperfections(a):
    """Per-imperfection (hard) and per-easy-mode results for text overwrites, from the
    per-image CSVs of the final model on validation and test."""
    fm, _ = final_model()
    sources = {"val": "results/val_model_run2_per_image.csv" if fm["chosen"] == "run2" else "results/val_model_per_image.csv",
               "test": "results/test_model_per_image.csv"}
    out = {"evidence_class": "SYNTHETIC; small groups (n per row shown); descriptive only, no CIs",
           "model": fm["chosen"], "detected_definition": "per-image TP pixels > 0 at the frozen pixel threshold"}
    for split, path in sources.items():
        p = resolve(path)
        if not p.exists():
            continue
        df = pd.read_csv(p)
        man = {r["sample_id"]: r for r in read_manifest(resolve("data") / "manifests" / f"{split}.jsonl")}
        df = df[df["manipulation_type"] == "text_overwrite"].copy()
        df["edit"] = df["sample_id"].map(lambda s: man[s]["edit"])
        df["detected"] = df["tp"] > 0
        df["imperfections"] = df["edit"].map(lambda e: e.get("imperfections") or [])
        df["easy_mode"] = df["edit"].map(lambda e: e.get("easy_mode"))
        df["size_change_rel"] = df["edit"].map(lambda e: abs(e["edit_font_size"] - e["orig_font_size"]) / e["orig_font_size"])
        df["mask_pixels"] = df["sample_id"].map(lambda s: man[s]["mask_positive_pixels"])
        df["field_kind"] = df["edit"].map(lambda e: e.get("field_kind"))

        def summ(g):
            tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
            return {"n": int(len(g)), "detected_frac": round(float(g["detected"].mean()), 4),
                    "mean_per_image_f1": round(float(g["f1"].mean()), 4),
                    "pooled_f1": round(float(2 * tp / max(2 * tp + fp + fn, 1)), 4),
                    "median_mask_pixels": float(g["mask_pixels"].median())}

        res = {"by_difficulty": {d: summ(g) for d, g in df.groupby("difficulty")}}
        hard = df[df["difficulty"] == "hard"]
        res["hard_by_imperfection"] = {}
        for imp in sorted({i for lst in hard["imperfections"] for i in lst}):
            has = hard["imperfections"].map(lambda l: imp in l)
            res["hard_by_imperfection"][imp] = {"with": summ(hard[has]), "without": summ(hard[~has])}
        hard_n = hard["imperfections"].map(len)
        res["hard_by_number_of_imperfections"] = {int(k): summ(g) for k, g in hard.groupby(hard_n)}
        easy = df[df["difficulty"] == "easy"]
        res["easy_by_mode"] = {m: summ(g) for m, g in easy.groupby("easy_mode")}
        res["by_field_kind"] = {k: summ(g) for k, g in df.groupby("field_kind")}
        q = df["mask_pixels"].median()
        res["by_mask_size"] = {"small (<= median)": summ(df[df["mask_pixels"] <= q]),
                               "large (> median)": summ(df[df["mask_pixels"] > q]), "median_mask_pixels": float(q)}
        res["easy_relative_size_change_median"] = round(float(easy["size_change_rel"].median()), 4)
        out[split] = res
    path = resolve("results/overwrite_imperfections.json")
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("wrote", path)


# --- examples ---------------------------------------------------------------------------------
def pick_examples(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Deterministic selection rules on the frozen TEST per-image results (ties by sample_id)."""
    df = df.sort_values("sample_id")
    tam, auth = df[df["label"] == 1], df[df["label"] == 0]
    ov = tam[tam["manipulation_type"] == "text_overwrite"]
    sp = tam[tam["manipulation_type"] == "splice"]
    picks = []

    def take(frame, n, why, by, asc):
        rows = frame.sort_values([by, "sample_id"], ascending=[asc, True])
        for sid in rows["sample_id"]:
            if len([p for p in picks if p[1] == why]) >= n:
                break
            if sid not in [p[0] for p in picks]:
                picks.append((sid, why))

    take(ov[(ov["difficulty"] == "hard") & (ov["tp"] == 0)], 2, "missed hard overwrite (largest mask)", "fn", False)
    take(ov[(ov["difficulty"] == "easy") & (ov["tp"] == 0)], 2, "missed easy overwrite (largest mask)", "fn", False)
    take(ov, 2, "best-localised overwrite", "f1", False)
    take(sp, 2, "worst-localised splice", "f1", True)
    take(sp, 1, "best-localised splice", "f1", False)
    take(auth, 2, "authentic page with the highest risk score", "score", False)
    take(tam, 1, "tampered page with the lowest risk score", "score", True)
    return picks


def examples(a):
    fm, frozen = final_model()
    thr, img_thr = float(frozen["pixel_threshold"]), frozen["image_threshold"]
    df = pd.read_csv(resolve("results/test_model_per_image.csv"))
    picks = pick_examples(df)
    pred = predictor(fm, frozen, a.threads)
    from docforge.preprocess import load_mask, load_rgb
    data_root = resolve("data")
    man = {r["sample_id"]: r for r in read_manifest(data_root / "manifests" / "test.jsonl")}
    fdir = resolve("results/failures")
    fdir.mkdir(parents=True, exist_ok=True)
    rows, panels = [], []
    for k, (sid, why) in enumerate(picks, 1):
        r = man[sid]
        img = load_rgb(data_root / r["image_path"])
        gt = load_mask(data_root / r["mask_path"])
        prob = pred(img)
        H, W = gt.shape
        if gt.any():
            ys, xs = np.nonzero(gt)
            cy, cx = int(ys.mean()), int(xs.mean())
        else:
            cy, cx = np.unravel_index(int(np.argmax(prob)), prob.shape)
        half = 220
        y0, x0 = int(np.clip(cy - half, 0, H - 2 * half)), int(np.clip(cx - half, 0, W - 2 * half))
        sl = (slice(y0, y0 + 2 * half), slice(x0, x0 + 2 * half))
        crop = img[sl].copy()
        heat = cv2.applyColorMap((prob[sl] * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        over = (0.55 * crop + 0.45 * heat).astype(np.uint8)
        for target, mask, col in ((crop, gt[sl], (0, 200, 0)), (over, gt[sl], (0, 200, 0)),
                                  (over, prob[sl] >= thr, (255, 0, 255))):
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(target, cnts, -1, col, 2)
        page = cv2.resize(img, (int(W * 440 / H), 440), interpolation=cv2.INTER_AREA)
        sc = 440 / H
        cv2.rectangle(page, (int(x0 * sc), int(y0 * sc)), (int((x0 + 2 * half) * sc), int((y0 + 2 * half) * sc)), (255, 0, 0), 2)
        # any predicted pixel inside a thumbnail pixel marks it (area average > 0)
        pmask = cv2.resize((prob >= thr).astype(np.float32), page.shape[1::-1], interpolation=cv2.INTER_AREA)
        page[pmask > 0] = (255, 0, 255)
        panel = np.concatenate([page, crop, over], axis=1)
        label = f"#{k} {sid} | {why} | F1 {df.set_index('sample_id').loc[sid, 'f1']:.2f} risk {df.set_index('sample_id').loc[sid, 'score']:.2f}"
        bar = np.full((28, panel.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        panel = np.concatenate([bar, panel], axis=0)
        cv2.imwrite(str(fdir / f"{k:02d}_{sid}.png"), panel[:, :, ::-1])
        panels.append(panel)
        e = r.get("edit") or {}
        row = df.set_index("sample_id").loc[sid]
        rows.append({"rank": k, "sample_id": sid, "selection_rule": why, "document_type": r["document_type"],
                     "manipulation_type": r["manipulation_type"], "difficulty": r.get("difficulty"),
                     "splice_size_class": r.get("splice_size_class"), "layout": r["layout"],
                     "processing": r["processing"], "mask_pixels": r.get("mask_positive_pixels"),
                     "per_image_f1": round(float(row["f1"]), 4), "tp": int(row["tp"]), "fp": int(row["fp"]),
                     "fn": int(row["fn"]), "risk_score": round(float(row["score"]), 4),
                     "flagged_at_frozen_image_threshold": bool(row["score"] >= img_thr),
                     "max_score_inside_mask": round(float(prob[gt].max()), 4) if gt.any() else None,
                     "page_max_score": round(float(prob.max()), 4),
                     "edit": {k2: e.get(k2) for k2 in ("field_kind", "old_value", "new_value", "easy_mode", "imperfections",
                                                       "orig_font_family", "edit_font_family", "orig_font_size",
                                                       "edit_font_size", "offset_px", "letter_spacing_px", "supersample",
                                                       "aliased")} if e else None,
                     "donor_source_id": r.get("donor_source_id"), "panel": f"results/failures/{k:02d}_{sid}.png"})
        print(f"[{k}/{len(picks)}] {sid}: {why}", flush=True)
    width = max(p.shape[1] for p in panels)
    sheet = np.concatenate([np.pad(p, ((0, 6), (0, width - p.shape[1]), (0, 0)), constant_values=255) for p in panels])
    fig = resolve("docs/figures")
    fig.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fig / "failure_examples.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])
    out = {"evidence_class": "SYNTHETIC test pages; frozen run, no changes; explanations are hypotheses",
           "model": fm["chosen"], "pixel_threshold": thr, "image_threshold": img_thr,
           "panel_legend": "left: page (red = crop, magenta = predicted pixels); middle: crop with ground truth "
                           "(green); right: score heatmap overlay with ground truth (green) and prediction (magenta)",
           "examples": rows}
    path = resolve("results/failure_examples.json")
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path, "and docs/figures/failure_examples.jpg")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("regions")
    r.add_argument("--split", default="val", choices=["val"], help="validation only")
    r.add_argument("--threads", type=int, default=12)
    sub.add_parser("imperfections")
    e = sub.add_parser("examples")
    e.add_argument("--threads", type=int, default=12)
    a = ap.parse_args()
    {"regions": regions, "imperfections": imperfections, "examples": examples}[a.cmd](a)


if __name__ == "__main__":
    main()
