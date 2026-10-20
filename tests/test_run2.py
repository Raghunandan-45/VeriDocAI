"""Run-2 training options (per-crop Dice, type-stratified validation subset, macro-F1
selection) and the guarantee that the default config still reproduces run 1."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch
import yaml

from research.config import load_config, resolve
from research.gen.dataset_hash import read_manifest
from research.train import bce_dice_loss, macro_selection, type_f1_curves, val_subset


def _two_crop_batch():
    """Crop 0: large edit (32x32 of 64x64), predicted perfectly. Crop 1: small edit (4x4),
    missed completely (confident 0). Batch Dice hides the miss; per-crop Dice does not."""
    y = torch.zeros(2, 1, 64, 64)
    y[0, :, :32, :32] = 1
    y[1, :, 10:14, 10:14] = 1
    logits = torch.full_like(y, -10.0)
    logits[0, :, :32, :32] = 10.0
    return logits, y


def test_batch_mode_is_the_run1_formula():
    logits, y = _two_crop_batch()
    p = torch.sigmoid(logits)
    expected = 1.0 - (2.0 * (p * y).sum() + 1.0) / (p.sum() + y.sum() + 1.0)
    _, _, dice = bce_dice_loss(logits, y)               # default mode
    _, _, dice_b = bce_dice_loss(logits, y, mode="batch")
    assert torch.equal(dice, expected) and torch.equal(dice_b, expected)


def test_per_crop_dice_weights_small_edits_equally():
    logits, y = _two_crop_batch()
    _, _, d_batch = bce_dice_loss(logits, y, mode="batch")
    _, _, d_crop = bce_dice_loss(logits, y, mode="per_crop")
    assert d_batch < 0.02                     # 16 missed pixels out of 1040: almost invisible
    assert 0.45 < d_crop < 0.5                # one of two positive crops fully missed
    # negative-only crops are excluded from the Dice average (BCE handles them)
    neg = torch.zeros(1, 1, 64, 64)
    logits3, y3 = torch.cat([logits, torch.full_like(neg, 5.0)]), torch.cat([y, neg])
    _, _, d3 = bce_dice_loss(logits3, y3, mode="per_crop")
    assert torch.allclose(d3, d_crop)


def test_per_crop_dice_all_negative_batch_has_gradient():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    loss, _, dice = bce_dice_loss(logits, torch.zeros(2, 1, 8, 8), mode="per_crop")
    assert dice.item() == 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_unknown_modes_raise(tiny_dataset):
    with pytest.raises(ValueError):
        bce_dice_loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4), mode="nope")
    with pytest.raises(ValueError):
        val_subset(tiny_dataset[1]["val"], 6, 1, mode="nope")


def test_stratified_subset_balanced_and_fixed(tiny_dataset):
    val = tiny_dataset[1]["val"]
    a = val_subset(val, 6, seed=1, mode="stratified_type")
    assert a == val_subset(val, 6, seed=1, mode="stratified_type")
    c = Counter(r["manipulation_type"] for r in a)
    assert c == {"none": 2, "splice": 2, "text_overwrite": 2}


def test_macro_selection_uses_one_shared_threshold():
    n_bins = 10
    # per-page histograms (counts per score bin): splice pages score high, overwrite pages mid
    pos = np.zeros((3, n_bins))
    neg = np.zeros((3, n_bins))
    pos[0, 9] = 100     # splice page: all tampered pixels in the top bin
    pos[1, 5] = 10      # overwrite page: tampered pixels at 0.5
    neg[2, 2] = 1000    # authentic page: everything low
    curves = type_f1_curves(pos, neg, ["splice", "text_overwrite", "none"])
    m = macro_selection(curves, n_bins)
    assert m["macro_threshold"] <= 0.5            # a splice-only threshold (0.9) would miss the overwrite
    assert m["f1_splice"] == pytest.approx(1.0) and m["f1_overwrite"] == pytest.approx(1.0)
    assert m["macro_f1"] == pytest.approx(1.0)
    assert macro_selection(type_f1_curves(pos[[0, 2]], neg[[0, 2]], ["splice", "none"]), n_bins)["macro_f1"] is None


def test_run2_config_differs_from_run1_only_in_approved_keys():
    r1, r2 = load_config("configs/train.yaml"), load_config("configs/train_run2.yaml")

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
        return out

    f1, f2 = flat(r1), flat(r2)
    assert f1.keys() == f2.keys()
    diff = {k for k in f1 if f1[k] != f2[k]}
    assert diff == {"output_dir", "train.epochs", "train.dice_mode", "val.subset_mode", "val.select_metric"}
    assert (f2["train.epochs"], f2["train.dice_mode"], f2["val.subset_mode"], f2["val.select_metric"]) == \
           (20, "per_crop", "stratified_type", "macro_f1")
    assert (f1["train.dice_mode"], f1["val.subset_mode"], f1["val.select_metric"]) == ("batch", "random", "pooled_f1")


def test_default_config_reproduces_run1():
    """configs/train.yaml (+ code defaults) gives the run-1 settings and, on the real data,
    exactly the run-1 validation subset recorded on Kaggle."""
    used_path = resolve("results/kaggle_run/config_used.yaml")
    if not used_path.exists():
        pytest.skip("run-1 config_used.yaml not present")
    used = yaml.safe_load(used_path.read_text())
    cfg = load_config("configs/train.yaml")
    for section in ("model", "train", "val", "smoke"):
        ran = used["train_config"][section]
        for k, v in cfg[section].items():
            if k in ran:
                assert ran[k] == v, f"{section}.{k}"
            else:   # keys added after run 1 must hold the value that reproduces run 1
                assert (section, k, v) in {("train", "dice_mode", "batch"), ("val", "subset_mode", "random"),
                                           ("val", "select_metric", "pooled_f1")}, f"{section}.{k}"
    assert cfg["seed"] == used["train_config"]["seed"]
    val_path = resolve(cfg["data_root"]) / "manifests" / "val.jsonl"
    if not val_path.exists():
        pytest.skip("full dataset not generated")
    sub = val_subset(read_manifest(val_path), cfg["val"]["subset_pages"], cfg["seed"], cfg["val"]["subset_mode"])
    assert [r["sample_id"] for r in sub] == used["val_subset_ids"]
