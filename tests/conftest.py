"""Shared fixtures.

`tiny_dataset` builds a 20-source dataset (10 per type) in a temp folder once per test
session, using the real generator code and configs/data.yaml with only the source counts
reduced. `dataset` runs dataset-level tests on the tiny dataset AND on the real generated
dataset in data/ (the latter is reported as SKIPPED if it has not been generated yet).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.config import load_config, resolve, with_overrides
from research.gen.build_dataset import build_dataset

TINY = {"sources.invoice": 10, "sources.marksheet": 10}


def read_manifests(root: Path) -> dict[str, list[dict]]:
    out = {}
    for split in ("train", "val", "test"):
        with open(root / "manifests" / f"{split}.jsonl") as f:
            out[split] = [json.loads(line) for line in f if line.strip()]
    return out


@pytest.fixture(scope="session")
def data_cfg() -> dict:
    return load_config("configs/data.yaml")


@pytest.fixture(scope="session")
def tiny_cfg(data_cfg) -> dict:
    return with_overrides(data_cfg, TINY)


@pytest.fixture(scope="session")
def tiny_dataset(tiny_cfg, tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_data")
    build_dataset(tiny_cfg, root, workers=4)
    return root, read_manifests(root)


@pytest.fixture(scope="session", params=["tiny", "full"])
def dataset(request, data_cfg, tiny_dataset):
    """(data_root, {split: records}) for the tiny test dataset and the real one."""
    if request.param == "tiny":
        return tiny_dataset
    root = resolve(data_cfg["paths"]["data_root"])
    if not (root / "manifests" / "train.jsonl").exists():
        pytest.skip("full dataset not generated yet (run research.gen.build_dataset)")
    return root, read_manifests(root)
