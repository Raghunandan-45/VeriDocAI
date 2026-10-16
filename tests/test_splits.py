"""Dataset tests 7-8: source-level split isolation and same-split splice donors."""
from __future__ import annotations

from collections import Counter, defaultdict

from research.gen.build_dataset import assign_splits, plan_sources


def test_no_source_or_variant_crosses_splits(dataset):
    root, manifests, _ = dataset
    split_of_source = defaultdict(set)
    for split, recs in manifests.items():
        for r in recs:
            assert r["split"] == split, f"{r['sample_id']} listed in {split} but says {r['split']}"
            assert r["image_path"].startswith(f"generated/{split}/")
            split_of_source[r["source_id"]].add(split)
    crossing = {s: sp for s, sp in split_of_source.items() if len(sp) > 1}
    assert not crossing, f"sources in several splits: {crossing}"
    # every source has all of its variants (authentic + text overwrite + splice)
    variants = defaultdict(set)
    for recs in manifests.values():
        for r in recs:
            variants[r["source_id"]].add(r["manipulation_type"])
    assert all(v == {"none", "text_overwrite", "splice"} for v in variants.values())


def test_splice_donor_in_same_split(dataset):
    root, manifests, _ = dataset
    split_of_source = {r["source_id"]: r["split"] for recs in manifests.values() for r in recs}
    n = 0
    for recs in manifests.values():
        for r in recs:
            if r["manipulation_type"] == "splice":
                d = r["donor_source_id"]
                assert d is not None and d != r["source_id"]
                assert split_of_source[d] == r["split"], f"{r['sample_id']}: donor {d} in other split"
                n += 1
    assert n > 0


def test_difficulty_and_splice_class_balanced_per_split(dataset):
    """Within each split x document type: text overwrites are 50/50 easy/hard and splices are
    50/50 field/block (exact for even group sizes; off by at most one otherwise)."""
    root, manifests, _ = dataset
    for split, recs in manifests.items():
        for doc in ("invoice", "marksheet"):
            c = Counter(r["difficulty"] for r in recs
                        if r["document_type"] == doc and r["manipulation_type"] == "text_overwrite")
            assert abs(c["easy"] - c["hard"]) <= 1, f"{split}/{doc}: {dict(c)}"
            c = Counter(r["splice_size_class"] for r in recs
                        if r["document_type"] == doc and r["manipulation_type"] == "splice")
            assert abs(c["field"] - c["block"]) <= 1 and set(c) == {"field", "block"}, f"{split}/{doc}: {dict(c)}"


def test_split_sizes_and_determinism(data_cfg):
    """Full config: 200 per type -> 140/30/30 per type, identical on every call."""
    sources = plan_sources(data_cfg)
    a, b = assign_splits(sources, data_cfg), assign_splits(sources, data_cfg)
    assert a == b
    for doc in ("invoice", "marksheet"):
        counts = defaultdict(int)
        for s in sources:
            if s["document_type"] == doc:
                counts[a[s["source_id"]]] += 1
        assert dict(counts) == {"train": 140, "val": 30, "test": 30}


def test_no_near_blank_splice_patch(dataset):
    """Every splice patch meets its class ink floor (block 0.05, field 0.03 by default)."""
    root, manifests, cfg = dataset
    sc = cfg["tamper"]["splice"]
    for recs in manifests.values():
        for r in recs:
            if r["manipulation_type"] == "splice":
                floor = sc[r["splice_size_class"]]["min_ink_density"]
                assert r["edit"]["donor_ink_density"] >= floor, f"{r['sample_id']}: near-blank patch"
