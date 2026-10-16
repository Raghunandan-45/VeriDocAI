"""Dataset counts table and contact sheet (image beside mask) for visual review.

    .venv/bin/python -m research.gen.report     # regenerate both from data/manifests
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from research.config import load_config, resolve  # noqa: E402

ROWS = [("none", None), ("text_overwrite", "easy"), ("text_overwrite", "hard"), ("splice", None),
        ("text_overwrite", "hard")]


def counts_table(by_split: dict[str, list[dict]]) -> pd.DataFrame:
    rows = [{"split": r["split"], "document_type": r["document_type"],
             "manipulation_type": r["manipulation_type"], "difficulty": r["difficulty"] or "n/a"}
            for recs in by_split.values() for r in recs]
    df = pd.DataFrame(rows)
    return (df.groupby(["split", "document_type", "manipulation_type", "difficulty"])
            .size().rename("count").reset_index())


def write_counts(by_split, path: Path) -> pd.DataFrame:
    t = counts_table(by_split)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(path, index=False)
    print(t.to_string(index=False))
    return t


def _zoom_box(rec, W, H):
    """Window around the edit (or a fixed upper-middle window for authentic pages)."""
    if rec["bbox_xyxy"] is None:
        cx, cy, w, h = W // 2, int(H * 0.35), 520, 260
    else:
        x0, y0, x1, y1 = rec["bbox_xyxy"]
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        w, h = max(360, int((x1 - x0) * 1.8)), max(180, int((y1 - y0) * 2.5))
    x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
    return x0, y0, min(W, x0 + w), min(H, y0 + h)


def _caption(r) -> str:
    p = r["processing"]
    s = f"{r['sample_id']} [{r['split']}]  {r['manipulation_type']}"
    if r["difficulty"]:
        s += f"/{r['difficulty']}"
    e = r["edit"] or {}
    if r["manipulation_type"] == "text_overwrite":
        s += f"\n'{e['old_value']}' -> '{e['new_value']}'  font {e['orig_font_family']} {e['orig_font_size']}px"
        s += f" -> {e['edit_font_family']} {e['edit_font_size']}px"
        if e["imperfections"]:
            s += f"  imperf={','.join(e['imperfections'])}"
    elif r["manipulation_type"] == "splice":
        s += f"\ndonor {r['donor_source_id']} box {r['donor_box_xyxy']}"
    else:
        s += f"\nfont {r['font_family']}"
    s += (f"\nJPEG q1={p['stage1_jpeg_quality']} q2={p['final_jpeg_quality']}  scale={p['resize_scale']}"
          f" ({p['resize_interp']})  blur={p['blur_sigma']}")
    return s


def pick_samples(by_split, seed: int) -> list[dict]:
    """Deterministic selection covering doc type x manipulation x difficulty x split."""
    rng = np.random.default_rng([seed, 777])
    splits = list(by_split)
    chosen, i = [], 0
    for doc in ("invoice", "marksheet"):
        for manip, diff in ROWS:
            split = splits[i % len(splits)]
            i += 1
            pool = [r for r in by_split[split] if r["document_type"] == doc and r["manipulation_type"] == manip
                    and r["difficulty"] == diff and r not in chosen]
            if not pool:  # fall back to any split
                pool = [r for recs in by_split.values() for r in recs if r["document_type"] == doc
                        and r["manipulation_type"] == manip and r["difficulty"] == diff and r not in chosen]
            if pool:
                chosen.append(pool[int(rng.integers(len(pool)))])
    return chosen


def contact_sheet(by_split, data_root: Path, out: Path, seed: int) -> list[dict]:
    """Rows: page thumbnail (edit box in red) | zoomed image | zoomed mask | zoomed overlay."""
    recs = pick_samples(by_split, seed)
    fig, axes = plt.subplots(len(recs), 4, figsize=(22, 4.2 * len(recs)),
                             gridspec_kw={"width_ratios": [0.75, 1.6, 1.6, 1.6]})
    for row, r in zip(np.atleast_2d(axes), recs):
        img = np.asarray(Image.open(Path(data_root) / r["image_path"]).convert("RGB"))
        mask = np.asarray(Image.open(Path(data_root) / r["mask_path"])) > 127
        H, W = mask.shape
        x0, y0, x1, y1 = _zoom_box(r, W, H)
        row[0].imshow(img)
        row[0].add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="red", lw=1.5))
        row[1].imshow(img[y0:y1, x0:x1], interpolation="nearest")
        row[2].imshow(mask[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ov = img[y0:y1, x0:x1].astype(np.float32).copy()
        mz = mask[y0:y1, x0:x1]
        ov[mz] = 0.55 * ov[mz] + 0.45 * np.array([255, 0, 0])
        row[3].imshow(ov.astype(np.uint8), interpolation="nearest")
        row[0].set_title(f"{W}x{H}", fontsize=9)
        row[1].set_title(_caption(r), fontsize=9, loc="left")
        row[2].set_title(f"mask ({int(mask.sum())} px positive on page)", fontsize=9)
        row[3].set_title("overlay (mask in red)", fontsize=9)
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle("VeriDoc synthetic dataset - contact sheet (SYNTHETIC data, for visual review)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=72)
    plt.close(fig)
    return recs


def main():
    from research.gen.dataset_hash import read_manifest
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-root", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    root = resolve(a.data_root or cfg["paths"]["data_root"])
    by_split = {s: read_manifest(root / "manifests" / f"{s}.jsonl") for s in ("train", "val", "test")}
    write_counts(by_split, resolve(cfg["paths"]["counts_csv"]))
    contact_sheet(by_split, root, resolve(cfg["paths"]["contact_sheet"]), cfg["seed"])


if __name__ == "__main__":
    main()
