"""Post-hoc perturbations for the quick robustness check (configs/robustness.yaml).

Each perturbation takes the decoded saved page and its mask and returns a perturbed
(image, mask, params) triple. Only `resize` changes the geometry, and it resizes the mask
with NEAREST so the two stay aligned. Parameters are deterministic per page:
rng([seed, perturbation id, crc32(sample_id)]).
"""
from __future__ import annotations

import zlib

import numpy as np

from research.gen.postprocess import gaussian_blur, jpeg_roundtrip, resize_pair

PERTURBATIONS = {"jpeg": 1, "resize": 2, "blur": 3}


def page_rng(seed: int, name: str, sample_id: str) -> np.random.Generator:
    return np.random.default_rng([seed, PERTURBATIONS[name], zlib.crc32(sample_id.encode())])


def perturb(img: np.ndarray, mask: np.ndarray, name: str, cfg: dict, sample_id: str):
    """Apply perturbation `name` with parameters from `cfg` (configs/robustness.yaml)."""
    if name not in PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; choose from {sorted(PERTURBATIONS)}")
    rng = page_rng(cfg["seed"], name, sample_id)
    spec = cfg[name]
    if name == "jpeg":
        q = int(rng.integers(spec["quality"][0], spec["quality"][1] + 1))
        return jpeg_roundtrip(img, q), mask, {"jpeg_quality": q}
    if name == "resize":
        s = round(float(rng.uniform(*spec["scale"])), 4)
        img2, mask2 = resize_pair(img, mask, s, spec["interp"])
        return img2, mask2, {"resize_scale": s}
    s = round(float(rng.uniform(*spec["sigma"])), 4)
    return gaussian_blur(img, s), mask, {"blur_sigma": s}
