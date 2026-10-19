"""Model construction / checkpoint I/O / prediction, loss, and the page-visit training data."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from docforge.model import (build_model, load_checkpoint, load_encoder_weights, make_predict_fn, predict_page,
                            save_checkpoint)
from research.sampler import CropSampler
from research.train import VisitDataset, bce_dice_loss, collate_visits, val_subset


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return build_model(encoder_weights=None).eval()


def test_predict_fn_shape_and_range(model):
    tiles = np.random.default_rng(0).integers(0, 256, (3, 256, 256, 3), dtype=np.uint8)
    p = make_predict_fn(model)(tiles)
    assert p.shape == (3, 256, 256) and p.dtype == np.float32
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_checkpoint_roundtrip_same_outputs(model, tmp_path):
    path = tmp_path / "ck.pt"
    save_checkpoint(path, model, epoch=3, train_config={"seed": 1})
    loaded, meta = load_checkpoint(path, "cpu")
    assert meta["epoch"] == 3 and meta["train_config"] == {"seed": 1}
    tiles = np.random.default_rng(1).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    np.testing.assert_array_equal(make_predict_fn(model)(tiles), make_predict_fn(loaded)(tiles))


def test_offline_encoder_weights_loader(tmp_path):
    """A torchvision-format ResNet-18 state dict (the Kaggle offline fallback) loads into the
    U-Net encoder exactly; fc.* keys are ignored."""
    import torchvision
    tv = torchvision.models.resnet18(weights=None)
    path = tmp_path / "resnet18.pth"
    torch.save(tv.state_dict(), path)
    m = build_model(encoder_weights=None, weights_path=path)
    for k, v in tv.state_dict().items():
        if not k.startswith("fc."):
            assert torch.equal(m.encoder.state_dict()[k], v), k
    bad = tmp_path / "bad.pth"
    torch.save({"conv1.weight": tv.state_dict()["conv1.weight"]}, bad)
    with pytest.raises(ValueError):
        load_encoder_weights(build_model(None), bad)


@pytest.mark.parametrize("h,w", [(300, 517), (2100, 260)])  # second one exercises the 2000 px cap
def test_predict_page_returns_original_size(model, h, w):
    img = np.random.default_rng(2).integers(0, 256, (h, w, 3), dtype=np.uint8)
    prob = predict_page(img, make_predict_fn(model), max_side=2000, tile=256, stride=256)
    assert prob.shape == (h, w) and np.isfinite(prob).all() and 0 <= prob.min() and prob.max() <= 1


def test_bce_dice_loss_behaviour():
    y = torch.zeros(2, 1, 8, 8)
    y[:, :, 2:6, 2:6] = 1
    good = (y * 2 - 1) * 10.0            # confident and correct logits
    bad = -good
    lg, _, _ = bce_dice_loss(good, y)
    lb, _, _ = bce_dice_loss(bad, y)
    assert lg < 0.01 and lb > 1.5


def test_sample_visit_proportions_and_single_page(tiny_dataset, data_cfg):
    root, manifests, _ = tiny_dataset
    s = CropSampler(manifests["train"], root, data_cfg["crops"], cache_pages=4)
    q, p_t = s.visit_probs()
    assert q == pytest.approx(0.75) and p_t == pytest.approx(2 / 3)
    rng = np.random.default_rng(9)
    crops = []
    for _ in range(150):
        v = s.sample_visit(rng, 4)
        assert len(v) == 4 and len({c.sample_id for c in v}) == 1   # one page per visit
        crops += v
    pos = np.mean([c.is_positive for c in crops])
    negs = [c for c in crops if not c.is_positive]
    from_auth = np.mean(["authentic" in c.sample_id for c in negs])
    assert abs(pos - 0.5) < 0.07, pos
    assert abs(from_auth - 0.5) < 0.1, from_auth
    assert all(c.mask.sum() >= 64 for c in crops if c.is_positive)
    assert all(c.mask.sum() == 0 for c in negs)


def test_visit_dataset_deterministic_and_shapes(tiny_dataset, data_cfg):
    root, manifests, _ = tiny_dataset
    ds = VisitDataset(manifests["train"], root, data_cfg["crops"], n_visits=6, crops_per_visit=4, seed=1, page_cache=2)
    ds.set_epoch(2)
    x1, y1 = ds[3]
    ds2 = VisitDataset(manifests["train"], root, data_cfg["crops"], n_visits=6, crops_per_visit=4, seed=1, page_cache=2)
    ds2.set_epoch(2)
    x2, y2 = ds2[3]
    assert torch.equal(x1, x2) and torch.equal(y1, y2)
    assert x1.shape == (4, 3, 256, 256) and y1.shape == (4, 1, 256, 256)
    ds2.set_epoch(3)
    assert not torch.equal(ds2[3][0], x1)                     # new epoch -> new crops
    xb, yb = collate_visits([ds[0], ds[1]])
    assert xb.shape == (8, 3, 256, 256) and set(torch.unique(yb).tolist()) <= {0.0, 1.0}


def test_val_subset_fixed_and_mixed(tiny_dataset):
    _, manifests, _ = tiny_dataset
    a = val_subset(manifests["val"], 6, seed=1)
    assert a == val_subset(manifests["val"], 6, seed=1)
    c = Counter(r["image_label"] for r in a)
    assert c[0] == 2 and c[1] == 4 and all(r["split"] == "val" for r in a)
