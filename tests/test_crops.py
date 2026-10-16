"""Dataset test 9: the training crop sampler follows the positive/negative rules."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from research.sampler import CropSampler


@pytest.fixture(scope="module")
def crops(tiny_dataset, data_cfg):
    root, manifests = tiny_dataset
    s = CropSampler(manifests["train"], root, data_cfg["crops"])
    rng = np.random.default_rng(123)
    return [s.sample(rng) for _ in range(600)], data_cfg["crops"]


def test_crop_shapes_and_binary_masks(crops):
    cs, cfg = crops
    for c in cs:
        assert c.image.shape == (cfg["size"], cfg["size"], 3) and c.image.dtype == np.uint8
        assert c.mask.shape == (cfg["size"], cfg["size"])
        assert set(np.unique(c.mask)) <= {0, 1}


def test_positive_and_negative_rules(crops):
    cs, cfg = crops
    for c in cs:
        if c.is_positive:
            assert int(c.mask.sum()) >= cfg["min_positive_pixels"]
        else:
            assert int(c.mask.sum()) == 0, "negative crop contains tampered pixels"
            assert c.ink_density >= cfg["min_ink_density"], "near-blank negative"
            if c.mode in ("neg_text_dense", "neg_near_edit"):
                assert c.ink_density >= cfg["text_dense_threshold"]


def test_class_and_mode_proportions(crops):
    cs, cfg = crops
    n = len(cs)
    pos = sum(c.is_positive for c in cs) / n
    assert abs(pos - cfg["p_positive"]) < 0.07, f"positive fraction {pos:.3f}"
    negs = [c for c in cs if not c.is_positive]
    modes = Counter(c.mode for c in negs)
    dense = (modes["neg_text_dense"] + modes["neg_near_edit"]) / len(negs)
    assert abs(dense - cfg["neg_text_dense_frac"]) < 0.08, f"text-dense fraction {dense:.3f}"
    assert modes["neg_near_edit"] > 0, "no hard negatives near edits"
    # negatives come from both authentic pages and untouched regions of tampered pages
    srcs = Counter("authentic" in c.sample_id for c in negs)
    assert srcs[True] > 0 and srcs[False] > 0


def test_edit_not_always_centred(crops):
    cs, cfg = crops
    offs = []
    for c in cs:
        if c.is_positive:
            ys, xs = np.nonzero(c.mask)
            offs.append((xs.mean() - cfg["size"] / 2, ys.mean() - cfg["size"] / 2))
    offs = np.array(offs)
    assert offs.std(axis=0).min() > 20, f"edit centroid offsets too concentrated: std {offs.std(axis=0)}"


def test_sampler_refuses_non_train_records(tiny_dataset, data_cfg):
    root, manifests = tiny_dataset
    for split in ("val", "test"):
        with pytest.raises(ValueError):
            CropSampler(manifests["train"] + manifests[split][:1], root, data_cfg["crops"])
