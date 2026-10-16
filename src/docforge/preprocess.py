"""Image loading and size handling shared by research evaluation and the serving API."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_rgb(src: str | Path | bytes) -> np.ndarray:
    """Decode a file path or raw bytes to an HxWx3 uint8 RGB array."""
    im = Image.open(io.BytesIO(src)) if isinstance(src, (bytes, bytearray)) else Image.open(src)
    return np.asarray(im.convert("RGB"))


def load_mask(path: str | Path) -> np.ndarray:
    """Binary mask (bool HxW) from a 0/255 PNG."""
    return np.asarray(Image.open(path)) > 127


def cap_long_side(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Downscale so the long side is <= max_side (INTER_AREA). Returns (image, scale);
    scale == 1.0 means the image was not changed. Never upscales."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA), scale


def resize_map(prob: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """Resize a float score map back to (H, W) with bilinear interpolation."""
    if prob.shape == tuple(shape_hw):
        return prob
    return cv2.resize(prob, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_LINEAR)


def imagenet_normalize(tiles: np.ndarray) -> np.ndarray:
    """uint8 NxHxWx3 RGB tiles -> float32 Nx3xHxW, ImageNet mean/std normalised."""
    x = tiles.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))
