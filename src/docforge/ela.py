"""Error Level Analysis (ELA) baseline.

Output is an "ELA anomaly score map" in [0, 1], NOT a probability:

    diff  = |image - JPEG_resave(image, quality)|          per pixel and channel
    d     = max over the 3 channels                         (combine = "max"; or "mean")
    d     = GaussianBlur(d, smooth_sigma)                  (optional; 0 disables)
    score = clip(d / norm_const, 0, 1)

Normalisation uses a FIXED constant from config (configs/eval.yaml `ela.norm_const`), not
per-image min-max: per-image min-max would stretch the pure noise of an authentic page to
the full [0, 1] range. The constant (10 grey levels) was set from the maximum smoothed
difference observed on 24 random TRAIN pages (9.5), so the score rarely saturates; it
does not change ranking-based metrics, only where the clip happens.

ELA is computed at the image's native resolution (resampling would destroy the JPEG
block structure it relies on), so the analysis-size cap is not applied to it.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def jpeg_resave(img: np.ndarray, quality: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), subsampling=2)
    return np.asarray(Image.open(buf).convert("RGB"))


def ela_map(img: np.ndarray, quality: int = 90, combine: str = "max", smooth_sigma: float = 2.0,
            norm_const: float = 10.0) -> np.ndarray:
    """ELA anomaly score map (HxW float32 in [0, 1]) for an HxWx3 uint8 RGB image."""
    diff = np.abs(img.astype(np.int16) - jpeg_resave(img, quality).astype(np.int16)).astype(np.float32)
    if combine == "max":
        d = diff.max(axis=2)
    elif combine == "mean":
        d = diff.mean(axis=2)
    else:
        raise ValueError(f"unknown combine {combine!r}")
    if smooth_sigma > 0:
        d = cv2.GaussianBlur(d, (0, 0), sigmaX=float(smooth_sigma), sigmaY=float(smooth_sigma),
                             borderType=cv2.BORDER_REFLECT)
    return np.clip(d / float(norm_const), 0.0, 1.0).astype(np.float32)
