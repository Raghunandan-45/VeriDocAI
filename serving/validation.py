"""Upload checks for the analysis endpoint. Everything happens in memory.

Order: size -> identify format from the bytes (not the filename or MIME type) -> width/height
from the header, BEFORE the pixels are decoded -> full decode (truncated or corrupt files fail
here) -> RGB array.
"""
from __future__ import annotations

import io
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError


class UploadError(Exception):
    """A rejected upload: HTTP status, short machine-readable code, human-readable message."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def decode_upload(data: bytes, cfg: dict) -> tuple[np.ndarray, dict]:
    """Validate and decode an uploaded JPEG/PNG. `cfg` is the `upload` section of
    configs/serve.yaml. Returns (HxWx3 uint8 RGB array, info) or raises UploadError."""
    if not data:
        raise UploadError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > cfg["max_bytes"]:
        raise UploadError(413, "file_too_large", f"The file is larger than {cfg['max_bytes'] // (1024 * 1024)} MB.")
    try:
        im = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise UploadError(415, "unsupported_or_malformed", "The file is not a readable JPEG or PNG image.")
    except Exception:
        raise UploadError(400, "malformed_image", "The image header could not be read.")
    if im.format not in cfg["formats"]:
        raise UploadError(415, "unsupported_format",
                          f"Only {' and '.join(cfg['formats'])} are accepted (got {im.format}).")
    w, h = im.size
    if max(w, h) > cfg["max_side"] or w * h > cfg["max_pixels"]:
        raise UploadError(422, "image_too_large",
                          f"Image is {w}x{h}; the limit is {cfg['max_side']} px per side and "
                          f"{cfg['max_pixels'] // 1_000_000} megapixels.")
    if min(w, h) < cfg["min_side"]:
        raise UploadError(422, "image_too_small", f"Image is {w}x{h}; each side must be at least {cfg['min_side']} px.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            im.load()
            rgb = np.asarray(im.convert("RGB"))
    except Exception:
        raise UploadError(400, "malformed_image", "The image data is truncated or corrupt.")
    if rgb.shape[:2] != (h, w):
        raise UploadError(400, "malformed_image", "Decoded size does not match the image header.")
    return rgb, {"format": im.format, "width": w, "height": h, "bytes": len(data)}
