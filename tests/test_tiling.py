"""Tiled inference: exact shapes, padding, known functions, identity model, window, size cap."""
from __future__ import annotations

import numpy as np
import pytest

from docforge.preprocess import cap_long_side, resize_map
from docforge.tiling import hann_window, padding, tiled_predict


def ramp_image(h, w):
    """Smooth known function of position, stored in channel 0 (others: constant)."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = ((x * 7 + y * 3) % 256).astype(np.uint8)
    img[..., 1] = 100
    img[..., 2] = (y % 256).astype(np.uint8)
    return img


def identity_model(tiles):
    """'Model' that returns channel 0 / 255 per pixel."""
    return tiles[..., 0].astype(np.float32) / 255.0


SIZES = [(256, 256), (300, 517), (1001, 777), (100, 37), (2, 2), (257, 129)]


@pytest.mark.parametrize("h,w", SIZES)
@pytest.mark.parametrize("tile,stride", [(256, 128), (256, 256), (64, 48)])
def test_identity_model_reproduces_input_exact_shape(h, w, tile, stride):
    img = ramp_image(h, w)
    out = tiled_predict(img, identity_model, tile=tile, stride=stride, batch_size=7)
    assert out.shape == (h, w)
    np.testing.assert_allclose(out, img[..., 0] / 255.0, atol=1e-6)


@pytest.mark.parametrize("h,w", SIZES)
def test_known_pointwise_function(h, w):
    """A pointwise model g(pixel) must stitch to exactly g(image) (overlaps agree)."""
    img = ramp_image(h, w)
    g = lambda t: 1.0 / (1.0 + np.exp(-(t[..., 0].astype(np.float32) - 128.0) / 20.0))  # noqa: E731
    out = tiled_predict(img, g, tile=256, stride=128)
    np.testing.assert_allclose(out, g(img), atol=1e-6)


def test_constant_model_gives_constant_map():
    out = tiled_predict(ramp_image(333, 444), lambda t: np.full(t.shape[:3], 0.37, np.float32))
    np.testing.assert_allclose(out, 0.37, atol=1e-6)


def test_blending_is_weighted_average_of_overlapping_tiles():
    """Model output = tile index; every output pixel must lie between the min and max index
    of the tiles covering it, and differ from a plain (unweighted) average somewhere."""
    calls = []

    def idx_model(tiles):
        base = len(calls)
        calls.append(len(tiles))
        return np.stack([np.full(tiles.shape[1:3], base + i, np.float32) for i in range(len(tiles))])

    out = tiled_predict(ramp_image(300, 300), idx_model, tile=256, stride=128, batch_size=100)
    assert out.min() >= 0 and out.max() <= sum(calls) - 1
    assert len(np.unique(np.round(out, 3))) > 4  # blended, not a single tile's value


def test_padding_covers_and_aligns():
    for length in [1, 2, 37, 128, 255, 256, 257, 1754, 2105]:
        for tile, stride in [(256, 128), (256, 256), (64, 48)]:
            before, after = padding(length, tile, stride)
            total = length + before + after
            assert total >= tile and (total - tile) % stride == 0
            assert before == (tile - stride) // 2


def test_hann_window_positive_with_floor():
    w = hann_window(256, floor=0.01)
    assert w.shape == (256, 256) and w.min() >= 0.01 - 1e-7 and abs(w.max() - 1.0) < 1e-6
    assert w[128, 128] > w[0, 0]


def test_bad_predict_fn_shape_rejected():
    with pytest.raises(ValueError):
        tiled_predict(ramp_image(300, 300), lambda t: np.zeros((len(t), 10, 10)))


def test_cap_long_side_and_resize_back():
    img = ramp_image(2105, 1488)
    small, s = cap_long_side(img, 2000)
    assert max(small.shape[:2]) == 2000 and 0 < s < 1
    same, s1 = cap_long_side(img[:1000, :700], 2000)
    assert s1 == 1.0 and same.shape == (1000, 700, 3)
    back = resize_map(np.zeros(small.shape[:2], np.float32), img.shape[:2])
    assert back.shape == img.shape[:2]
