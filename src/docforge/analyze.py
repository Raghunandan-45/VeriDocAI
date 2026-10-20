"""Turn a full-page score map into the outputs the serving API returns.

The page score map comes from docforge.model.predict_page, the same function the
evaluation harness uses. Thresholding uses docforge.scoring.binarize, and the image score
uses the rule frozen on validation, so evaluation and serving cannot drift apart.

No verdict is derived here. The image threshold in the frozen config is deliberately not
used: on the synthetic test split it flagged 92% of authentic pages (results/test_model.json).
"""
from __future__ import annotations

import cv2
import numpy as np

from docforge.preprocess import cap_long_side
from docforge.scoring import binarize, top_k_mean


def image_risk_score(prob: np.ndarray, mask: np.ndarray, rule: str, top_k: int) -> float:
    """Image-level risk score with the frozen rule (same definitions as research/evaluate.py).
    Not a calibrated probability."""
    if rule == "max":
        return float(prob.max())
    if rule == "top_k_mean":
        return top_k_mean(prob, top_k)
    if rule == "frac_above":
        return float(mask.mean())
    raise ValueError(f"unknown score rule {rule!r}")


def mask_regions(mask: np.ndarray, prob: np.ndarray, limit: int) -> tuple[list[dict], int]:
    """Bounding boxes of the 8-connected components of the predicted mask, highest peak score
    first. Display only: no area filter or merging (those would be post-processing that was
    not selected on validation). Returns (at most `limit` regions, total number)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    regions = []
    for c in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[c])
        scores = prob[y:y + h, x:x + w][labels[y:y + h, x:x + w] == c]
        regions.append({"bbox_xyxy": [x, y, x + w, y + h], "area_px": area,
                        "max_score": round(float(scores.max()), 4), "mean_score": round(float(scores.mean()), 4)})
    regions.sort(key=lambda r: (-r["max_score"], -r["area_px"]))
    return regions[:limit], len(regions)


def heatmap_rgb(prob: np.ndarray) -> np.ndarray:
    """Score map (0..1) as an RGB colour image (OpenCV JET: blue = low, red = high)."""
    u8 = np.clip(prob * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def overlay_rgb(img: np.ndarray, prob: np.ndarray, mask: np.ndarray, regions: list[dict],
                alpha: float) -> np.ndarray:
    """Page blended with the heatmap where the score is high, predicted pixels outlined in
    magenta and region boxes in red."""
    heat = heatmap_rgb(prob).astype(np.float32)
    a = (alpha * np.clip(prob, 0, 1))[..., None]
    out = (img.astype(np.float32) * (1 - a) + heat * a).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thick = max(1, round(max(img.shape[:2]) / 800))
    cv2.drawContours(out, contours, -1, (255, 0, 255), thick)
    for r in regions:
        x0, y0, x1, y1 = r["bbox_xyxy"]
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (220, 0, 0), thick)
    return out


def analyze_page(img: np.ndarray, prob: np.ndarray, frozen: dict, max_regions: int = 50,
                 display_max_side: int = 1600, overlay_alpha: float = 0.45) -> dict:
    """Outputs for one page from its full-resolution score map `prob` (same H x W as `img`).

    Boxes and areas are in original image pixels. The heatmap and overlay images are
    downscaled to at most `display_max_side` on the long side to keep responses small."""
    if prob.shape != img.shape[:2]:
        raise ValueError(f"score map {prob.shape} does not match image {img.shape[:2]}")
    mask = binarize(prob, frozen["pixel_threshold"], frozen["n_bins"])
    regions, n_regions = mask_regions(mask, prob, max_regions)
    overlay = overlay_rgb(img, prob, mask, regions, overlay_alpha)
    overlay_small, scale = cap_long_side(overlay, display_max_side)
    heat_small, _ = cap_long_side(heatmap_rgb(prob), display_max_side)
    return {"risk_score": image_risk_score(prob, mask, frozen["score_rule"], frozen["top_k"]),
            "regions": regions, "n_regions": n_regions,
            "predicted_area_fraction": float(mask.mean()),
            "heatmap": heat_small, "overlay": overlay_small, "display_scale": scale}
