"""Config loading and repo-relative path resolution shared by research scripts."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    """Load a YAML config file. Relative paths are resolved against the repo root."""
    return yaml.safe_load(resolve(path).read_text())


def resolve(path: str | Path) -> Path:
    """Return `path` as an absolute Path; relative paths are taken from the repo root
    (not the current working directory), so scripts behave the same from any folder."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a deep copy of `cfg` with dotted-key overrides applied,
    e.g. {"sources.invoice": 10, "build.workers": 0}."""
    out = copy.deepcopy(cfg)
    for key, value in overrides.items():
        node = out
        *parents, leaf = key.split(".")
        for k in parents:
            node = node[k]
        node[leaf] = value
    return out
