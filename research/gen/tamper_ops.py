"""Tampering operations: text overwrite and region splice.

Both functions work in render coordinates on the (stage-1 compressed) clean page and return
the edited image, a boolean mask of the INTENDED edit region, and metadata. The mask is:
  - text overwrite: union of the cover rectangle and the new text's ink box;
  - splice: the pasted patch footprint.
Compression traces, blur halos and anything outside these rectangles are not labelled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from research.gen import fake
from research.gen.fake import pick
from research.gen.fonts import FontRegistry, composite, render_text_alpha
from research.gen.templates import RESULTS, TextField


@dataclass
class EditResult:
    image: np.ndarray            # HxWx3 uint8, edited page
    mask: np.ndarray             # HxW bool, intended edit region
    info: dict                   # JSON-serialisable edit metadata
    covered: np.ndarray | None = None  # text overwrite only: page after covering, before new text


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    return (max(0, int(x0)), max(0, int(y0)), min(width, int(x1)), min(height, int(y1)))


def ink_density(rgb: np.ndarray, gray_threshold: int = 128) -> float:
    """Fraction of pixels darker than `gray_threshold` (a cheap 'how much text' measure)."""
    if rgb.size == 0:
        return 0.0
    gray = rgb.mean(axis=2) if rgb.ndim == 3 else rgb
    return float((gray < gray_threshold).mean())


def local_background(img: np.ndarray, box, ring: int):
    """Robust colour statistics of a `ring`-pixel band around `box`.

    Returns (median RGB, robust std) where std = 1.4826 * median absolute deviation, so that
    a table line or a neighbouring character inside the ring barely affects the estimate.
    """
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    ox0, oy0, ox1, oy1 = clamp_box((x0 - ring, y0 - ring, x1 + ring, y1 + ring), W, H)
    outer = img[oy0:oy1, ox0:ox1].reshape(-1, 3).astype(np.float32)
    inner = np.zeros((oy1 - oy0, ox1 - ox0), bool)
    inner[y0 - oy0:y1 - oy0, x0 - ox0:x1 - ox0] = True
    band = outer[~inner.reshape(-1)]
    med = np.median(band, axis=0)
    std = 1.4826 * np.median(np.abs(band - med), axis=0)
    return med, float(np.mean(std))


def cover_region(img: np.ndarray, box, ring: int, rng) -> dict:
    """Paint `box` (in place) with the local background colour plus matching noise."""
    med, std = local_background(img, box, ring)
    std = min(std, 4.0)
    x0, y0, x1, y1 = box
    fill = med + rng.normal(0.0, std, size=(y1 - y0, x1 - x0, 1)) if std > 0 else np.broadcast_to(med, (y1 - y0, x1 - x0, 3))
    img[y0:y1, x0:x1] = np.clip(np.rint(fill), 0, 255).astype(np.uint8)
    return {"fill_rgb": [round(float(v), 1) for v in med], "fill_noise_std": round(std, 3)}


def new_value(field: TextField, rng) -> str:
    """A plausible replacement value of the same kind as `field.text` (never identical)."""
    old = field.text
    for _ in range(20):
        k = field.kind
        if k == "amount":
            factor = float(rng.uniform(0.2, 0.8)) if rng.random() < 0.5 else float(rng.uniform(1.3, 4.0))
            val = round(fake.parse_amount(old) * factor, 2)
            new = fake.amount(val if val >= 1 else float(rng.uniform(10, 1000)))
        elif k == "int":
            lo, hi = field.meta.get("lo", 0), field.meta.get("hi", 99)
            new = str(int(rng.integers(lo, max(hi, lo + 1) + 1)))
        elif k == "date":
            new = fake.random_date(rng, field.meta.get("fmt", "%d/%m/%Y"))
        elif k == "name":
            new = fake.person_name(rng)
        elif k == "grade":
            new = pick(rng, fake.GRADES)
        elif k == "result":
            new = pick(rng, RESULTS)
        elif k == "percent":
            new = f"{float(rng.uniform(35, 99.9)):.2f}%"
        else:  # id / text
            new = fake.perturb_digits(old, rng)
        if new != old:
            return new
    return fake.perturb_digits(old, rng, 1)


def _sign(rng) -> float:
    return 1.0 if rng.random() < 0.5 else -1.0


def text_overwrite(img: np.ndarray, field: TextField, difficulty: str, tcfg: dict,
                   fonts: FontRegistry, families: list[str], rng) -> EditResult:
    """Cover `field` with local background and draw a new value at the same anchor.

    easy: different font family and/or a noticeably different size.
    hard: same family, similar size, plus a random non-empty subset of small imperfections
          (sub-pixel/1-2 px offset, ink colour shift, different anti-aliasing, letter spacing).
    """
    H, W = img.shape[:2]
    out = img.copy()
    pad = int(rng.integers(tcfg["cover_pad_px"][0], tcfg["cover_pad_px"][1] + 1))
    x0, y0, x1, y1 = field.bbox
    cover = clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), W, H)
    fill = cover_region(out, cover, tcfg["cover_ring_px"], rng)
    covered = out.copy()

    text = new_value(field, rng)
    family, size, color = field.family, field.size, tuple(field.color)
    offset, spacing, supersample, aliased = (0.0, 0.0), 0.0, 1, False
    imperfections: list[str] = []
    easy_mode = None
    if difficulty == "easy":
        easy_mode = pick(rng, ["family", "size", "both"])
        if easy_mode in ("family", "both"):
            family = pick(rng, [f for f in families if f != field.family])
        if easy_mode in ("size", "both"):
            e = tcfg["easy"]
            lo, hi = e["size_ratio_small"] if rng.random() < 0.5 else e["size_ratio_large"]
            new_size = int(round(field.size * float(rng.uniform(lo, hi))))
            if abs(new_size - field.size) < 2:
                new_size = field.size + (2 if new_size >= field.size else -2)
            size = max(8, new_size)
    else:
        h = tcfg["hard"]
        size = field.size + int(pick(rng, h["size_delta_px"]))
        names = ["offset", "ink_shift", "antialias", "letter_spacing"]
        imperfections = [names[i] for i in sorted(rng.choice(4, size=int(rng.integers(1, 5)), replace=False))]
        if len(text) < 2 and "letter_spacing" in imperfections:
            # spacing has no effect on a single character; keep at least one real imperfection
            imperfections.remove("letter_spacing")
            imperfections = imperfections or ["offset"]
        if "offset" in imperfections:
            dx = _sign(rng) * float(rng.uniform(*h["offset_px"]))
            dy = _sign(rng) * float(rng.uniform(0.0, h["offset_px"][1])) if rng.random() < 0.5 else 0.0
            offset = (round(dx, 3), round(dy, 3))
        if "ink_shift" in imperfections:
            chans = rng.choice(3, size=int(rng.integers(1, 4)), replace=False)
            c = list(color)
            for ch in chans:
                c[int(ch)] = int(np.clip(c[int(ch)] + _sign(rng) * int(rng.integers(h["ink_shift"][0], h["ink_shift"][1] + 1)), 0, 255))
            if tuple(c) == tuple(color):  # clipped back to the same colour: shift upwards instead
                c[int(chans[0])] = min(255, c[int(chans[0])] + h["ink_shift"][0])
            color = tuple(c)
        if "antialias" in imperfections:
            if rng.random() < 0.3:
                aliased = True
            else:
                supersample = int(pick(rng, h["supersample"]))
        if "letter_spacing" in imperfections:
            spacing = round(_sign(rng) * float(rng.uniform(*h["letter_spacing_px"])), 3)

    ta = render_text_alpha(text, fonts.path(family, field.bold), size, field.xy, field.anchor,
                           letter_spacing=spacing, supersample=supersample, offset=offset, aliased=aliased)
    composite(out, ta, color)
    text_box = clamp_box(ta.bbox, W, H)

    mask = np.zeros((H, W), bool)
    mask[cover[1]:cover[3], cover[0]:cover[2]] = True
    mask[text_box[1]:text_box[3], text_box[0]:text_box[2]] = True
    info = {
        "field_name": field.name, "field_kind": field.kind,
        "old_value": field.text, "new_value": text,
        "orig_font_family": field.family, "orig_font_size": field.size,
        "edit_font_family": family, "edit_font_size": size, "bold": field.bold,
        "orig_color": list(field.color), "edit_color": list(color),
        "easy_mode": easy_mode, "imperfections": imperfections,
        "offset_px": list(offset), "letter_spacing_px": spacing,
        "supersample": supersample, "aliased": aliased,
        "cover_box_render_xyxy": list(cover), "text_box_render_xyxy": list(text_box), **fill,
    }
    return EditResult(out, mask, info, covered)


def clean_rerender(covered: np.ndarray, field: TextField, text: str, fonts: FontRegistry) -> np.ndarray:
    """Draw `text` exactly like the original field (same font, size, colour, position,
    default anti-aliasing) on the covered page. Used to prove hard edits are not identical
    to a clean render (test 10)."""
    out = covered.copy()
    ta = render_text_alpha(text, fonts.path(field.family, field.bold), field.size, field.xy, field.anchor)
    composite(out, ta, field.color)
    return out


def paper_color(img: np.ndarray) -> np.ndarray:
    """Per-channel median of the page (subsampled). Most of a document page is paper, so the
    median is a robust estimate of the paper colour even with text, tables and bands."""
    return np.median(img[::4, ::4].reshape(-1, 3), axis=0).astype(np.float32)


def _block_box(donor: np.ndarray, scfg: dict, rng):
    """Large block: a random box (size from config) anywhere inside the page margins."""
    m, b = scfg["margin_px"], scfg["block"]
    w = int(rng.integers(b["width_px"][0], b["width_px"][1] + 1))
    h = int(rng.integers(b["height_px"][0], b["height_px"][1] + 1))
    dx = int(rng.integers(m, donor.shape[1] - m - w))
    dy = int(rng.integers(m, donor.shape[0] - m - h))
    return (dx, dy, dx + w, dy + h), {}


def _field_box(donor: np.ndarray, donor_fields: list[TextField], scfg: dict, rng):
    """Field-sized patch: starts at a random donor text field and spans 1-3 text lines."""
    f_cfg = scfg["field"]
    H, W = donor.shape[:2]
    cands = [f for f in donor_fields if f.bbox[2] - f.bbox[0] >= 15 and f.name != "logo"]
    f = cands[int(rng.integers(len(cands)))]
    n_lines = int(rng.integers(f_cfg["lines"][0], f_cfg["lines"][1] + 1))
    pad = int(rng.integers(f_cfg["pad_px"][0], f_cfg["pad_px"][1] + 1))
    extra = int(rng.integers(f_cfg["extra_width_px"][0], f_cfg["extra_width_px"][1] + 1))
    pitch = int(round(f.size * f_cfg["line_pitch_factor"]))
    x0, y0, x1, y1 = f.bbox
    box = clamp_box((x0 - pad, y0 - pad, x1 + pad + extra, y1 + (n_lines - 1) * pitch + pad), W, H)
    return box, {"donor_field_name": f.name, "lines": n_lines}


def donor_box(donor, donor_fields, size_class, scfg, rng, gray_threshold=128):
    """Resample a donor box of the given class until its ink density reaches the class floor
    (`block.min_ink_density` / `field.min_ink_density`), so no patch is near-blank paper.
    Raises if `max_tries` boxes all fail (never silently accepts a blank patch)."""
    floor = scfg[size_class]["min_ink_density"]
    for _ in range(scfg["max_tries"]):
        if size_class == "field":
            box, extra = _field_box(donor, donor_fields, scfg, rng)
        else:
            box, extra = _block_box(donor, scfg, rng)
        dens = ink_density(donor[box[1]:box[3], box[0]:box[2]], gray_threshold)
        if dens >= floor:
            return box, dens, extra
    raise RuntimeError(f"no {size_class} donor box with ink density >= {floor} in {scfg['max_tries']} tries")


def splice(target: np.ndarray, donor: np.ndarray, donor_fields: list[TextField], size_class: str,
           scfg: dict, rng, gray_threshold: int = 128) -> EditResult:
    """Paste a rectangular patch of `donor` at a random location of `target`.

    size_class "block": large random block containing ink; "field": 1-3 text lines taken
    from a donor text field. With tint_match, the patch is shifted by the difference of
    paper colours (target - donor) plus a small uniform residual, so it is not trivially
    visible as a differently tinted rectangle.
    """
    H, W = target.shape[:2]
    m = scfg["margin_px"]
    (dx0, dy0, dx1, dy1), dens, extra = donor_box(donor, donor_fields, size_class, scfg, rng, gray_threshold)
    w, h = dx1 - dx0, dy1 - dy0
    patch = donor[dy0:dy1, dx0:dx1].astype(np.float32)
    shift = np.zeros(3, np.float32)
    if scfg["tint_match"]:
        r = float(scfg["tint_residual"])
        shift = paper_color(target) - paper_color(donor) + rng.uniform(-r, r, 3).astype(np.float32)
        patch = patch + shift
    tx = int(rng.integers(m, W - m - w))
    ty = int(rng.integers(m, H - m - h))
    out = target.copy()
    out[ty:ty + h, tx:tx + w] = np.clip(np.rint(patch), 0, 255).astype(np.uint8)
    mask = np.zeros((H, W), bool)
    mask[ty:ty + h, tx:tx + w] = True
    info = {"size_class": size_class, "donor_box_xyxy": [dx0, dy0, dx1, dy1],
            "dest_box_render_xyxy": [tx, ty, tx + w, ty + h],
            "donor_ink_density": round(dens, 4),
            "tint_shift_rgb": [round(float(v), 2) for v in shift], **extra}
    return EditResult(out, mask, info)
