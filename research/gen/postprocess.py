"""Post-processing applied IDENTICALLY to authentic and tampered pages.

Pipeline (see README): render -> stage-1 JPEG of clean page (optional, compression.double)
-> tamper -> resize -> optional Gaussian blur -> final JPEG (the saved image file).
Masks follow the geometry only: NEAREST resize then re-binarisation; never blurred.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

INTERP = {"bilinear": Image.Resampling.BILINEAR, "lanczos": Image.Resampling.LANCZOS}


def encode_jpeg(img: np.ndarray, quality: int) -> bytes:
    """Encode an RGB uint8 array as baseline JPEG (Pillow defaults, 4:2:0 subsampling)."""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=int(quality), subsampling=2, optimize=False)
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def jpeg_roundtrip(img: np.ndarray, quality: int) -> np.ndarray:
    """Compress and decompress in memory (used for the stage-1 save of the clean page)."""
    return decode_image(encode_jpeg(img, quality))


def resize_pair(img: np.ndarray, mask: np.ndarray, scale: float, interp: str):
    """Resize image (bilinear/Lanczos) and mask (NEAREST, re-binarised) to the same size."""
    H, W = img.shape[:2]
    size = (max(1, int(round(W * scale))), max(1, int(round(H * scale))))
    img2 = np.asarray(Image.fromarray(img).resize(size, INTERP[interp]))
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    return img2, np.asarray(m) > 127


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma),
                            borderType=cv2.BORDER_REFLECT)


def sample_post_params(rng, cfg: dict) -> dict:
    """Draw resize / blur / final JPEG parameters for one sample from config ranges."""
    r, b, c = cfg["resize"], cfg["blur"], cfg["compression"]
    blur_applied = bool(rng.random() < b["prob"])
    return {
        "resize_scale": round(float(rng.uniform(*r["scale"])), 4),
        "resize_interp": str(r["interpolations"][int(rng.integers(len(r["interpolations"])))]),
        "blur_applied": blur_applied,
        "blur_sigma": round(float(rng.uniform(*b["sigma"])), 4) if blur_applied else 0.0,
        "final_jpeg_quality": int(rng.integers(c["final_quality"][0], c["final_quality"][1] + 1)),
    }


def apply_post(img: np.ndarray, mask: np.ndarray, params: dict):
    """Resize + optional blur. Returns (image before final JPEG, final mask)."""
    img2, mask2 = resize_pair(img, mask, params["resize_scale"], params["resize_interp"])
    if params["blur_applied"]:
        img2 = gaussian_blur(img2, params["blur_sigma"])
    return img2, mask2
