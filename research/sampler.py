"""Training crop sampler (TRAIN split only).

Rules (configs/data.yaml `crops`):
- 256x256 crops at native resolution of the saved (post-processed) page.
- Positive crop: at least `min_positive_pixels` tampered pixels. The crop is placed so a
  random tampered pixel lands at a random position inside it (edit not always centred).
- Negative crop: ZERO tampered pixels. `neg_text_dense_frac` of negatives must have ink
  density >= `text_dense_threshold`; the rest are uniform but not near-blank
  (ink density >= `min_ink_density`). Negatives come from authentic pages and from untouched
  regions of tampered pages; on tampered pages, text-dense negatives are taken near the edit
  with probability `near_edit_prob` (hard negatives).
- Augmentation: optional small scale jitter applied identically to image and mask. No flips.

The sampler is plain numpy; a torch Dataset wraps it in research/train.py (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass
class Crop:
    image: np.ndarray   # size x size x 3 uint8
    mask: np.ndarray    # size x size uint8 in {0, 1}
    is_positive: bool
    mode: str           # positive | neg_text_dense | neg_near_edit | neg_uniform
    sample_id: str
    xy: tuple[int, int]  # top-left of the source window in page coordinates
    window: int          # source window side (differs from size when scale jitter is applied)
    ink_density: float


class PageCache:
    """Loads a page once and keeps image, mask, and integral images of ink and mask."""

    def __init__(self, data_root: Path, ink_gray_threshold: int, max_pages: int = 64):
        self.root = Path(data_root)
        self.thr = ink_gray_threshold
        self.get = lru_cache(maxsize=max_pages)(self._load)

    def _load(self, image_path: str, mask_path: str):
        img = np.asarray(Image.open(self.root / image_path).convert("RGB"))
        mask = (np.asarray(Image.open(self.root / mask_path)) > 127).astype(np.uint8)
        ink = (img.mean(axis=2) < self.thr).astype(np.int32)
        ink_ii = cv2.integral(ink.astype(np.uint8), sdepth=cv2.CV_32S)
        mask_ii = cv2.integral(mask, sdepth=cv2.CV_32S)
        return img, mask, ink_ii, mask_ii


def box_sum(ii: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """Sum over the window [y:y+h, x:x+w] from an integral image (OpenCV layout)."""
    return int(ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x])


class CropSampler:
    def __init__(self, records: list[dict], data_root: Path, crops_cfg: dict, cache_pages: int = 64):
        bad = sorted({r["split"] for r in records if r["split"] != "train"})
        if bad:
            raise ValueError(f"CropSampler only accepts TRAIN records; got splits {bad}")
        if not records:
            raise ValueError("no records")
        self.cfg = crops_cfg
        self.size = int(crops_cfg["size"])
        self.records = records
        self.tampered = [r for r in records if r["image_label"] == 1]
        self.authentic = [r for r in records if r["image_label"] == 0]
        self.cache = PageCache(data_root, crops_cfg["ink_gray_threshold"], cache_pages)

    # -- helpers ---------------------------------------------------------------------------
    def _page(self, r):
        return self.cache.get(r["image_path"], r["mask_path"])

    def _window(self, rng) -> int:
        """Source window side: `size`, or size/scale when scale jitter is drawn."""
        a = self.cfg["aug"]
        if rng.random() < a["scale_prob"]:
            return int(round(self.size / float(rng.uniform(*a["scale_range"]))))
        return self.size

    def _extract(self, img, mask, x, y, win):
        """Cut the window and resize to `size` (image: area/linear, mask: nearest)."""
        ci, cm = img[y:y + win, x:x + win], mask[y:y + win, x:x + win]
        if win != self.size:
            interp = cv2.INTER_AREA if win > self.size else cv2.INTER_LINEAR
            ci = cv2.resize(ci, (self.size, self.size), interpolation=interp)
            cm = cv2.resize(cm, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return np.ascontiguousarray(ci), np.ascontiguousarray(cm)

    def _density(self, ink_ii, x, y, win) -> float:
        return box_sum(ink_ii, x, y, win, win) / float(win * win)

    # -- public ----------------------------------------------------------------------------
    def sample(self, rng: np.random.Generator) -> Crop:
        """Draw one crop: positive with probability p_positive, else negative."""
        for _ in range(100):
            crop = self._positive(rng) if rng.random() < self.cfg["p_positive"] else self._negative(rng)
            if crop is not None:
                return crop
        raise RuntimeError("crop sampler failed 100 times in a row; check crop config")

    def _positive(self, rng) -> Crop | None:
        if not self.tampered:
            return None
        r = self.tampered[int(rng.integers(len(self.tampered)))]
        img, mask, ink_ii, mask_ii = self._page(r)
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        need = self.cfg["min_positive_pixels"]
        for _ in range(self.cfg["max_tries"]):
            win = self._window(rng)
            if win > min(H, W):
                continue
            j = int(rng.integers(len(xs)))
            # place the chosen tampered pixel at a random position inside the window
            x = int(np.clip(xs[j] - rng.integers(0, win), 0, W - win))
            y = int(np.clip(ys[j] - rng.integers(0, win), 0, H - win))
            if box_sum(mask_ii, x, y, win, win) < need:
                continue
            ci, cm = self._extract(img, mask, x, y, win)
            if int(cm.sum()) < need:   # re-check after scale jitter
                continue
            return Crop(ci, cm, True, "positive", r["sample_id"], (x, y), win, self._density(ink_ii, x, y, win))
        return None

    def _negative(self, rng) -> Crop | None:
        c = self.cfg
        text_dense = rng.random() < c["neg_text_dense_frac"]
        use_auth = rng.random() < c["neg_from_authentic_frac"] or not self.tampered
        pool = self.authentic if (use_auth and self.authentic) else self.tampered
        r = pool[int(rng.integers(len(pool)))]
        img, mask, ink_ii, mask_ii = self._page(r)
        H, W = mask.shape
        near = text_dense and r["image_label"] == 1 and rng.random() < c["near_edit_prob"]
        mode = "neg_near_edit" if near else ("neg_text_dense" if text_dense else "neg_uniform")
        min_dens = c["text_dense_threshold"] if text_dense else c["min_ink_density"]
        for _ in range(c["max_tries"]):
            win = self._window(rng)
            if win > min(H, W):
                continue
            if near:
                bx0, by0, bx1, by1 = r["bbox_xyxy"]
                mg = c["near_edit_margin_px"]
                x = int(rng.integers(max(0, bx0 - mg - win), max(1, min(W - win, bx1 + mg))))
                y = int(rng.integers(max(0, by0 - mg - win), max(1, min(H - win, by1 + mg))))
                x, y = min(x, W - win), min(y, H - win)
            else:
                x = int(rng.integers(0, W - win + 1))
                y = int(rng.integers(0, H - win + 1))
            if box_sum(mask_ii, x, y, win, win) != 0:
                continue
            dens = self._density(ink_ii, x, y, win)
            if dens < min_dens:
                continue
            ci, cm = self._extract(img, mask, x, y, win)
            return Crop(ci, cm, False, mode, r["sample_id"], (x, y), win, dens)
        return None
