"""Thresholding, pixel/image metrics, operating point and bootstrap confidence intervals.

Pixel scores are binned into `n_bins` equal bins on [0, 1]. A threshold t means "predict
tampered if bin(score) >= round(t * n_bins)", i.e. score >= t up to bin resolution. Using the
same `binarize` everywhere (grid search, evaluation, serving) keeps the frozen threshold
exactly reproducible. Per image we only keep two histograms (scores of ground-truth positive
and negative pixels), which is enough to compute every pixel metric at every threshold
without keeping full-page score maps in memory.

Conventions:
- Pooled pixel metrics sum TP/FP/FN over all pages (authentic pages contribute FP only).
- Per-image pixel metrics are averaged over TAMPERED pages only (F1 is undefined when a page
  has no positive pixels); precision is 0 when nothing is predicted.
- Image-level outputs are "risk scores", not calibrated probabilities.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

SCORE_RULES = ("max", "top_k_mean", "frac_above")


# --- binning / thresholds ------------------------------------------------------------------
def bin_index(prob: np.ndarray, n_bins: int) -> np.ndarray:
    return np.clip((np.asarray(prob, np.float64) * n_bins).astype(np.int64), 0, n_bins - 1)


def threshold_bin(threshold: float, n_bins: int) -> int:
    return int(round(threshold * n_bins))


def binarize(prob: np.ndarray, threshold: float, n_bins: int) -> np.ndarray:
    """Binary mask: True where bin(prob) >= bin(threshold). Use this everywhere."""
    return bin_index(prob, n_bins) >= threshold_bin(threshold, n_bins)


def pixel_hists(prob: np.ndarray, gt: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Histograms (length n_bins) of scores of ground-truth positive and negative pixels."""
    b = bin_index(prob, n_bins).ravel()
    g = np.asarray(gt, bool).ravel()
    return (np.bincount(b[g], minlength=n_bins).astype(np.int64),
            np.bincount(b[~g], minlength=n_bins).astype(np.int64))


def counts_at(pos: np.ndarray, neg: np.ndarray, k: int) -> tuple[int, int, int]:
    """(TP, FP, FN) when predicting bins >= k."""
    return int(pos[k:].sum()), int(neg[k:].sum()), int(pos[:k].sum())


def prf_iou(tp, fp, fn) -> dict:
    """Precision, recall, F1, IoU (0 where the denominator is 0). Works on arrays too."""
    tp, fp, fn = (np.asarray(v, np.float64) for v in (tp, fp, fn))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        iou = np.where(tp + fp + fn > 0, tp / (tp + fp + fn), 0.0)
    out = {"precision": p, "recall": r, "f1": f1, "iou": iou}
    return {k: float(v) if v.ndim == 0 else v for k, v in out.items()}


def f1_curve(pos_sum: np.ndarray, neg_sum: np.ndarray) -> np.ndarray:
    """Pooled F1 for every threshold bin k = 0..n_bins-1 (vectorised)."""
    tp = np.cumsum(pos_sum[::-1])[::-1]
    fp = np.cumsum(neg_sum[::-1])[::-1]
    fn = pos_sum.sum() - tp
    return prf_iou(tp, fp, fn)["f1"]


def select_pixel_threshold(pos: list[np.ndarray], neg: list[np.ndarray], n_bins: int) -> dict:
    """Grid search (all bin edges) for the threshold maximising POOLED pixel F1.
    Call only with VALIDATION data. Ties go to the lowest threshold."""
    curve = f1_curve(np.sum(pos, axis=0), np.sum(neg, axis=0))
    k = int(np.argmax(curve))
    return {"pixel_threshold": k / n_bins, "n_bins": n_bins, "val_pooled_f1_at_threshold": float(curve[k]),
            "f1_curve_every_10_bins": [round(float(v), 5) for v in curve[::10]]}


# --- image-level scores ----------------------------------------------------------------------
def top_k_mean(prob: np.ndarray, k: int) -> float:
    flat = np.asarray(prob, np.float32).ravel()
    k = min(k, flat.size)
    return float(np.partition(flat, flat.size - k)[flat.size - k:].mean())


def frac_above(pos: np.ndarray, neg: np.ndarray, k: int) -> float:
    """Fraction of all page pixels in bins >= k (rule c)."""
    total = pos.sum() + neg.sum()
    return float((pos[k:].sum() + neg[k:].sum()) / total) if total else 0.0


def roc_auc(labels, scores) -> float | None:
    from sklearn.metrics import roc_auc_score  # lazy: the serving image does not need scikit-learn

    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


# --- operating point ----------------------------------------------------------------------
def select_operating_point(scores, labels, target_recall: float, min_pos: int = 30, min_neg: int = 30,
                           min_unique: int = 10) -> dict:
    """On VALIDATION data: the highest image-score threshold whose recall on tampered pages is
    >= target_recall (predict tampered if score >= threshold). Reports whether the operating
    point is reliable; if not, `image_threshold` is None and `reasons` says why."""
    s, y = np.asarray(scores, np.float64), np.asarray(labels, int)
    pos, neg = np.sort(s[y == 1])[::-1], s[y == 0]
    reasons = []
    if len(pos) < min_pos:
        reasons.append(f"only {len(pos)} tampered validation pages (< {min_pos})")
    if len(neg) < min_neg:
        reasons.append(f"only {len(neg)} authentic validation pages (< {min_neg})")
    if len(np.unique(s)) < min_unique:
        reasons.append(f"degenerate scores: {len(np.unique(s))} unique values (< {min_unique})")
    out = {"target_recall": target_recall, "n_tampered": int(len(pos)), "n_authentic": int(len(neg))}
    if len(pos) == 0:
        return {**out, "image_threshold": None, "reliable": False, "reasons": reasons or ["no tampered pages"]}
    t = float(pos[math.ceil(target_recall * len(pos)) - 1])
    if len(neg) and t <= neg.min() and t <= pos.min():
        reasons.append("threshold flags every validation page as tampered")
    out.update(val_recall=float((pos >= t).mean()), val_fpr=float((neg >= t).mean()) if len(neg) else None,
               authentic_ties_at_threshold=int((neg == t).sum()), reasons=reasons, reliable=not reasons,
               image_threshold=t if not reasons else None, candidate_threshold=t)
    return out


def recall_fpr_at(scores, labels, threshold: float) -> dict:
    s, y = np.asarray(scores, np.float64), np.asarray(labels, int)
    pred = s >= threshold
    return {"recall": float(pred[y == 1].mean()) if (y == 1).any() else None,
            "fpr": float(pred[y == 0].mean()) if (y == 0).any() else None}


# --- bootstrap ------------------------------------------------------------------------------
def bootstrap_ci(n_items: int, stat: Callable[[np.ndarray], float | None], n_boot: int = 1000,
                 seed: int = 0, level: float = 0.95) -> dict:
    """Percentile CI from resampling items (images) with replacement. Resamples where the
    statistic is undefined (e.g. AUC with one class) are skipped and counted."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        v = stat(rng.integers(0, n_items, n_items))
        if v is not None:
            vals.append(v)
    if not vals:
        return {"lo": None, "hi": None, "n_valid": 0}
    a = (1 - level) / 2
    return {"lo": float(np.quantile(vals, a)), "hi": float(np.quantile(vals, 1 - a)), "n_valid": len(vals)}
