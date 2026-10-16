"""Tiled full-page inference with overlap blending.

1. Reflect-pad the page: (tile - stride) // 2 on the top/left so every page pixel falls in
   the high-weight centre of some tile, and enough on the bottom/right that the tiles
   exactly cover the padded page.
2. Cut tiles of size `tile` every `stride` pixels and run `predict_fn` on batches of them.
3. Blend overlapping predictions with a 2-D Hann window plus a small positive floor (so edge
   pixels of a tile still count a little and the weight sum is never zero), divide by the
   accumulated weights, and crop the padding off again.

`predict_fn(tiles_uint8_NxTxTx3) -> NxTxT float` must already return probabilities
(the model wrapper applies the sigmoid per tile).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

PredictFn = Callable[[np.ndarray], np.ndarray]


def hann_window(tile: int, floor: float = 0.01) -> np.ndarray:
    """2-D Hann window (outer product), rescaled to [floor, 1]."""
    if tile == 1:
        return np.ones((1, 1), np.float32)
    n = np.arange(tile)
    h = 0.5 - 0.5 * np.cos(2 * np.pi * (n + 0.5) / tile)  # symmetric, strictly positive
    w = np.outer(h, h)
    w = w / w.max()
    return (floor + (1.0 - floor) * w).astype(np.float32)


def padding(length: int, tile: int, stride: int) -> tuple[int, int]:
    """(before, after) padding for one axis so tiles start at 0, stride, ... and the last
    tile ends exactly at the padded length."""
    before = (tile - stride) // 2
    need = max(length + 2 * before, tile)
    n_steps = int(np.ceil((need - tile) / stride))
    total = tile + n_steps * stride
    return before, total - length - before


def tiled_predict(img: np.ndarray, predict_fn: PredictFn, tile: int = 256, stride: int = 128,
                  batch_size: int = 16, floor: float = 0.01) -> np.ndarray:
    """Stitched HxW float32 map for an HxWx3 uint8 image (exactly the input's H and W)."""
    if not 0 < stride <= tile:
        raise ValueError("need 0 < stride <= tile")
    H, W = img.shape[:2]
    (pt, pb), (pl, pr) = padding(H, tile, stride), padding(W, tile, stride)
    padded = np.pad(img, ((pt, pb), (pl, pr), (0, 0)), mode="reflect")
    PH, PW = padded.shape[:2]
    win = hann_window(tile, floor)
    acc = np.zeros((PH, PW), np.float64)
    wsum = np.zeros((PH, PW), np.float64)
    coords = [(y, x) for y in range(0, PH - tile + 1, stride) for x in range(0, PW - tile + 1, stride)]
    for i in range(0, len(coords), batch_size):
        batch = coords[i:i + batch_size]
        tiles = np.stack([padded[y:y + tile, x:x + tile] for y, x in batch])
        probs = np.asarray(predict_fn(tiles), dtype=np.float64)
        if probs.shape != (len(batch), tile, tile):
            raise ValueError(f"predict_fn returned {probs.shape}, expected {(len(batch), tile, tile)}")
        for (y, x), p in zip(batch, probs):
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    out = acc / wsum
    return out[pt:pt + H, pl:pl + W].astype(np.float32)
