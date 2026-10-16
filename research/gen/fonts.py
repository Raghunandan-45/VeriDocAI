"""Font registry and the single text-rendering routine used by the generator.

All text (original document text AND tampered text) goes through `render_text_alpha` +
`composite`, so a "clean re-render" and a tampered render differ only in the parameters we
deliberately change (font, size, offset, spacing, anti-aliasing, colour).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from research.config import resolve


class FontRegistry:
    """Maps a family name to its vendored regular/bold TTF files (config `fonts`)."""

    def __init__(self, fonts_cfg: dict):
        root = resolve(fonts_cfg["dir"])
        self.paths = {
            fam: {"regular": str(root / files[0]), "bold": str(root / files[1])}
            for fam, files in fonts_cfg["families"].items()
        }
        for fam in self.paths:
            for style, p in self.paths[fam].items():
                if not Path(p).is_file():
                    raise FileNotFoundError(f"font file missing for {fam} {style}: {p}")

    def path(self, family: str, bold: bool = False) -> str:
        return self.paths[family]["bold" if bold else "regular"]


@lru_cache(maxsize=512)
def get_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Cached FreeType font (loading a TTF is slow relative to drawing)."""
    return ImageFont.truetype(path, size)


@dataclass
class TextAlpha:
    """Coverage (0..1) of rendered text, placed at (x0, y0) in page coordinates."""
    alpha: np.ndarray  # float32 HxW
    x0: int
    y0: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Tight xyxy box (exclusive x1, y1) of pixels with non-zero coverage."""
        ys, xs = np.nonzero(self.alpha > 0)
        if len(xs) == 0:
            return (self.x0, self.y0, self.x0, self.y0)
        return (self.x0 + int(xs.min()), self.y0 + int(ys.min()),
                self.x0 + int(xs.max()) + 1, self.y0 + int(ys.max()) + 1)


def render_text_alpha(text: str, font_path: str, size: int, xy: tuple[float, float],
                      anchor: str = "la", letter_spacing: float = 0.0, supersample: int = 1,
                      offset: tuple[float, float] = (0.0, 0.0), aliased: bool = False) -> TextAlpha:
    """Render `text` into a local coverage patch.

    Args:
        xy: anchor point in page pixels; `anchor` is a Pillow anchor ("la", "ra", "ma").
        letter_spacing: extra pixels added after each character (0 = normal spacing).
        supersample: render at Nx resolution and box-downsample; changes anti-aliasing and
            allows sub-pixel `offset`. 1 = normal Pillow rendering.
        aliased: disable anti-aliasing (Pillow fontmode "1").
    """
    s = int(supersample)
    font = get_font(font_path, size * s)
    x = (xy[0] + offset[0]) * s
    y = (xy[1] + offset[1]) * s
    total_w = font.getlength(text) + letter_spacing * s * max(len(text) - 1, 0)
    margin = 4 * s + size * s
    # Low-res patch origin (integers), so the hi-res canvas aligns with the page pixel grid.
    ox = int(np.floor((x - total_w) / s)) - margin // s - 2
    oy = int(np.floor(y / s)) - 2 * size - 4
    w = int(np.ceil(2 * total_w / s)) + 2 * (margin // s) + 8
    h = 4 * size + 8
    canvas = Image.new("L", (w * s, h * s), 0)
    draw = ImageDraw.Draw(canvas)
    draw.fontmode = "1" if aliased else "L"
    cx, cy = x - ox * s, y - oy * s
    if letter_spacing == 0:
        draw.text((cx, cy), text, font=font, fill=255, anchor=anchor)
    else:
        # Draw character by character; kerning-aware advances from getlength(prefix).
        h_anchor, v_anchor = anchor[0], anchor[1]
        start = {"l": cx, "m": cx - total_w / 2, "r": cx - total_w}[h_anchor]
        for i, ch in enumerate(text):
            px = start + font.getlength(text[:i]) + i * letter_spacing * s
            draw.text((px, cy), ch, font=font, fill=255, anchor="l" + v_anchor)
    a = np.asarray(canvas, dtype=np.float32) / 255.0
    if s > 1:
        a = a.reshape(h, s, w, s).mean(axis=(1, 3))
    # Crop to the non-zero region to keep patches small.
    ys, xs = np.nonzero(a > 0)
    if len(xs) == 0:
        return TextAlpha(np.zeros((0, 0), np.float32), ox, oy)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return TextAlpha(a[y0:y1, x0:x1].copy(), ox + int(x0), oy + int(y0))


def composite(page: np.ndarray, ta: TextAlpha, color: tuple[int, int, int]) -> None:
    """Alpha-blend `color` onto `page` (HxWx3 uint8, modified in place) using coverage `ta`."""
    H, W = page.shape[:2]
    h, w = ta.alpha.shape
    x0, y0 = max(ta.x0, 0), max(ta.y0, 0)
    x1, y1 = min(ta.x0 + w, W), min(ta.y0 + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    a = ta.alpha[y0 - ta.y0:y1 - ta.y0, x0 - ta.x0:x1 - ta.x0, None]
    region = page[y0:y1, x0:x1].astype(np.float32)
    out = region * (1.0 - a) + np.asarray(color, np.float32) * a
    page[y0:y1, x0:x1] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
