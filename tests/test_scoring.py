"""Metrics on hand-built masks, threshold/rule/operating-point selection, bootstrap, ELA and
the test-set guard of research/evaluate.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from docforge.ela import ela_map
from docforge.scoring import (binarize, bootstrap_ci, counts_at, frac_above, pixel_hists, prf_iou,
                              recall_fpr_at, roc_auc, select_operating_point, select_pixel_threshold,
                              threshold_bin, top_k_mean)
from research import evaluate as ev
from research.config import load_config

N_BINS = 1000


# --- hand-computed pixel metrics ------------------------------------------------------------
def test_metrics_on_hand_built_masks():
    gt = np.zeros((4, 4), bool)
    gt[0, 0:4] = True                       # 4 positives
    pred = np.zeros((4, 4), bool)
    pred[0, 0:3] = True                     # 3 TP
    pred[3, 0:2] = True                     # 2 FP ; FN = 1
    prob = pred.astype(np.float32)          # scores 1.0 / 0.0
    pos, neg = pixel_hists(prob, gt, N_BINS)
    tp, fp, fn = counts_at(pos, neg, threshold_bin(0.5, N_BINS))
    assert (tp, fp, fn) == (3, 2, 1)
    m = prf_iou(tp, fp, fn)
    assert m["precision"] == pytest.approx(3 / 5)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(2 * 0.6 * 0.75 / (0.6 + 0.75))  # 0.6667
    assert m["iou"] == pytest.approx(3 / 6)


def test_pooled_vs_per_image_and_authentic_page():
    # image A: tp=3 fp=2 fn=1 ; image B (authentic): fp=4 ; image C: tp=1 fp=0 fn=3
    tp, fp, fn = np.array([3, 0, 1]), np.array([2, 4, 0]), np.array([1, 0, 3])
    pooled = prf_iou(tp.sum(), fp.sum(), fn.sum())
    assert pooled["f1"] == pytest.approx(2 * 4 / (2 * 4 + 6 + 4))      # 8/18
    per = prf_iou(tp[[0, 2]], fp[[0, 2]], fn[[0, 2]])                   # tampered pages only
    assert per["f1"].mean() == pytest.approx((2 / 3 + 2 / 5) / 2)
    assert prf_iou(0, 0, 0)["f1"] == 0.0                                 # no division error


def test_hist_metrics_match_direct_computation():
    rng = np.random.default_rng(0)
    prob = rng.random((60, 70)).astype(np.float32)
    gt = rng.random((60, 70)) < 0.3
    pos, neg = pixel_hists(prob, gt, N_BINS)
    for t in [0.0, 0.123, 0.5, 0.9, 0.999]:
        pred = binarize(prob, t, N_BINS)
        direct = (int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum()))
        assert counts_at(pos, neg, threshold_bin(t, N_BINS)) == direct


def test_threshold_grid_finds_best_threshold():
    # positives score 0.70-0.80, negatives 0.10-0.65 -> any threshold in (0.65, 0.70] is perfect
    rng = np.random.default_rng(1)
    gt = np.zeros(2000, bool)
    gt[:500] = True
    prob = np.where(gt, rng.uniform(0.70, 0.80, 2000), rng.uniform(0.10, 0.65, 2000))
    pos, neg = pixel_hists(prob, gt, N_BINS)
    sel = select_pixel_threshold([pos], [neg], N_BINS)
    assert sel["val_pooled_f1_at_threshold"] == pytest.approx(1.0)
    # ties go to the LOWEST perfect threshold: one bin above the highest negative score
    lowest_perfect = (np.floor(prob[~gt].max() * N_BINS) + 1) / N_BINS
    assert sel["pixel_threshold"] == pytest.approx(lowest_perfect)
    assert 0.65 <= sel["pixel_threshold"] <= 0.70


# --- image scores and AUC -----------------------------------------------------------------
def test_image_score_rules():
    prob = np.zeros((10, 100), np.float32)
    prob[0, :5] = [0.9, 0.8, 0.7, 0.6, 0.5]
    assert prob.max() == pytest.approx(0.9)
    assert top_k_mean(prob, 3) == pytest.approx(0.8)
    pos, neg = pixel_hists(prob, np.zeros_like(prob, bool), N_BINS)
    assert frac_above(pos, neg, threshold_bin(0.65, N_BINS)) == pytest.approx(3 / 1000)


def test_roc_auc_hand_example():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
    assert roc_auc([1, 1], [0.2, 0.3]) is None


# --- operating point ---------------------------------------------------------------------
def test_operating_point_reaches_target_recall_on_given_data():
    rng = np.random.default_rng(2)
    pos, neg = rng.uniform(0.3, 1.0, 60), rng.uniform(0.0, 0.6, 60)
    scores, labels = np.r_[pos, neg], np.r_[np.ones(60), np.zeros(60)]
    op = select_operating_point(scores, labels, 0.9)
    assert op["reliable"] and op["image_threshold"] is not None
    assert op["val_recall"] >= 0.9
    # highest threshold with recall >= 0.9: raising it to the next tampered score breaks it
    higher = np.sort(pos)[np.sort(pos) > op["image_threshold"]]
    assert recall_fpr_at(scores, labels, higher.min())["recall"] < 0.9
    assert op["val_fpr"] == pytest.approx(float((neg >= op["image_threshold"]).mean()))


def test_operating_point_unreliable_cases():
    small = select_operating_point([0.9, 0.8, 0.1], [1, 1, 0], 0.9)
    assert not small["reliable"] and small["image_threshold"] is None
    assert any("tampered" in r for r in small["reasons"])
    flat = select_operating_point(np.full(80, 0.5), np.r_[np.ones(40), np.zeros(40)], 0.9)
    assert not flat["reliable"] and any("degenerate" in r for r in flat["reasons"])


# --- bootstrap -------------------------------------------------------------------------------
def test_bootstrap_deterministic_and_brackets_estimate():
    x = np.random.default_rng(3).normal(5.0, 1.0, 200)
    f = lambda b: float(x[b].mean())  # noqa: E731
    a, b = bootstrap_ci(len(x), f, 500, seed=7), bootstrap_ci(len(x), f, 500, seed=7)
    assert a == b and a["lo"] < x.mean() < a["hi"] and a["n_valid"] == 500
    skip = bootstrap_ci(10, lambda b: None, 50)
    assert skip["n_valid"] == 0 and skip["lo"] is None


# --- ELA -----------------------------------------------------------------------------------
def test_ela_map_range_and_signal():
    """Region never JPEG-compressed (fresh noise) on a JPEG page has a higher ELA score."""
    from docforge.ela import jpeg_resave
    rng = np.random.default_rng(4)
    page = jpeg_resave(rng.integers(150, 256, (256, 256, 3)).astype(np.uint8), 75)
    page = page.copy()
    page[64:128, 64:192] = rng.integers(0, 256, (64, 128, 3))
    m = ela_map(page, quality=90, smooth_sigma=2.0, norm_const=10.0)
    assert m.shape == (256, 256) and m.dtype == np.float32 and 0 <= m.min() and m.max() <= 1
    assert m[70:122, 70:186].mean() > 2 * np.delete(m, np.s_[56:136], axis=0).mean()
    assert np.array_equal(m, ela_map(page, quality=90, smooth_sigma=2.0, norm_const=10.0))


# --- test-set guard --------------------------------------------------------------------------
@pytest.fixture
def eval_cfg():
    return load_config("configs/eval.yaml")


@pytest.mark.parametrize("split", ["test", "test_unseen", "test_robust"])
def test_guard_refuses_test_split_without_frozen_config(split, eval_cfg):
    with pytest.raises(ev.TestSetGuardError):
        ev.check_guard(split, None, "ela", eval_cfg)


def test_guard_refuses_frozen_config_not_selected_on_val(eval_cfg):
    bad = {"selected_on": "test", "method": "ela", "ela": eval_cfg["ela"]}
    with pytest.raises(ev.TestSetGuardError):
        ev.check_guard("test", bad, "ela", eval_cfg)
    other = {"selected_on": "val", "method": "ela", "ela": {**eval_cfg["ela"], "quality": 75}}
    with pytest.raises(ev.TestSetGuardError):
        ev.check_guard("test", other, "ela", eval_cfg)
    ev.check_guard("test", {"selected_on": "val", "method": "ela", "ela": eval_cfg["ela"]}, "ela", eval_cfg)
    ev.check_guard("val", None, "ela", eval_cfg)  # validation may select


def test_stride_study_is_validation_only(eval_cfg):
    frozen = {"selected_on": "val", "method": "model", "checkpoint_sha256": "abc", "stride": 128, "max_side": 2000}
    info = {"checkpoint_sha256": "abc", "stride": 256, "max_side": 2000}
    with pytest.raises(ev.TestSetGuardError):           # a different stride is refused normally
        ev.check_guard("val", frozen, "model", eval_cfg, info)
    ev.check_guard("val", frozen, "model", eval_cfg, info, stride_study=True)
    for split in ("test", "test_unseen", "test_robust"):
        with pytest.raises(ev.TestSetGuardError):
            ev.check_guard(split, frozen, "model", eval_cfg, info, stride_study=True)
    with pytest.raises(ev.TestSetGuardError):           # needs a frozen config
        ev.check_guard("val", None, "model", eval_cfg, stride_study=True)
    with pytest.raises(ev.TestSetGuardError):           # still checks the checkpoint
        ev.check_guard("val", frozen, "model", eval_cfg, {**info, "checkpoint_sha256": "x"}, stride_study=True)


def test_cli_refuses_test_before_touching_data(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("data was read before the guard ran")
    monkeypatch.setattr(ev, "read_manifest", boom)
    monkeypatch.setattr(ev, "collect", boom)
    with pytest.raises(ev.TestSetGuardError):
        ev.run(["--method", "ela", "--split", "test"])


def _fake_stats(n_pos, n_neg, seed):
    rng = np.random.default_rng(seed)
    stats = []
    for i in range(n_pos + n_neg):
        lab = int(i < n_pos)
        pos = np.zeros(N_BINS, np.int64)
        neg = rng.integers(0, 5, N_BINS).astype(np.int64)
        if lab:
            pos[600:800] = rng.integers(1, 5, 200)
        stats.append({"sample_id": f"s{i}", "label": lab, "pos": pos, "neg": neg,
                      "max": float(rng.uniform(0.5, 1.0) if lab else rng.uniform(0.0, 0.8)),
                      "top_k_mean": float(rng.random())})
    return stats


def test_test_split_uses_frozen_values_and_never_selects(monkeypatch, tmp_path, eval_cfg):
    """With --frozen-config on a test split, the selection routines must never run, and the
    reported operating point is the frozen one."""
    recs = [{"sample_id": f"s{i}", "image_label": int(i < 40), "manipulation_type": "splice" if i < 40 else "none",
             "difficulty": None, "splice_size_class": "block" if i < 40 else None, "document_type": "invoice"}
            for i in range(80)]
    stats = _fake_stats(40, 40, 5)
    monkeypatch.setattr(ev, "read_manifest", lambda p: recs)
    monkeypatch.setattr(ev, "collect", lambda *a, **k: stats)

    def no_select(*a, **k):
        raise AssertionError("selection called on a test split")
    monkeypatch.setattr(ev, "select_on_validation", no_select)
    monkeypatch.setattr(ev, "select_pixel_threshold", no_select)
    monkeypatch.setattr(ev, "select_operating_point", no_select)
    frozen = {"method": "ela", "selected_on": "val", "pixel_threshold": 0.55, "n_bins": N_BINS,
              "score_rule": "max", "image_threshold": 0.77, "target_recall": 0.9, "ela": eval_cfg["ela"]}
    fz = tmp_path / "frozen.json"
    fz.write_text(json.dumps(frozen))
    out = tmp_path / "test_ela.json"
    res = ev.run(["--method", "ela", "--split", "test", "--frozen-config", str(fz), "--out", str(out)])
    assert res["frozen_config"]["image_threshold"] == 0.77
    s = np.array([st["max"] for st in stats])
    y = np.array([st["label"] for st in stats])
    expect = recall_fpr_at(s, y, 0.77)
    got = res["metrics"]["overall"]["at_frozen_image_threshold"]
    assert got["recall"] == pytest.approx(expect["recall"]) and got["fpr"] == pytest.approx(expect["fpr"])
    assert "selection" not in res


def test_operating_point_selected_from_validation_data_only(monkeypatch, tmp_path, eval_cfg):
    """On val, the frozen image threshold must equal the operating point computed from the
    validation scores alone (and the frozen file records selected_on = 'val')."""
    recs = [{"sample_id": f"s{i}", "image_label": int(i < 40), "manipulation_type": "splice" if i < 40 else "none",
             "difficulty": None, "splice_size_class": None, "document_type": "invoice"} for i in range(80)]
    stats = _fake_stats(40, 40, 6)
    monkeypatch.setattr(ev, "read_manifest", lambda p: recs)
    monkeypatch.setattr(ev, "collect", lambda *a, **k: stats)
    fz, out = tmp_path / "frozen.json", tmp_path / "val.json"
    res = ev.run(["--method", "ela", "--split", "val", "--out", str(out), "--frozen-out", str(fz)])
    frozen = json.loads(fz.read_text())
    assert frozen["selected_on"] == "val"
    rule = res["selection"]["score_rule"]
    k = threshold_bin(frozen["pixel_threshold"], N_BINS)
    scores = ev.image_scores(stats, rule, k)
    op = select_operating_point(scores, [s["label"] for s in stats], eval_cfg["operating_point"]["target_recall"])
    assert frozen["image_threshold"] == pytest.approx(op["image_threshold"])
