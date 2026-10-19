"""Path-detection helpers of research/kaggle_run.py, tested on fake /kaggle/input layouts.

The notebook itself cannot be imported here (it needs a GPU kernel), so the helper
functions are extracted from its source with `ast` and executed in isolation."""
from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from research.config import resolve

HELPERS = ("find_dirs_containing", "show_input_tree", "input_roots")


@pytest.fixture(scope="module")
def helpers():
    tree = ast.parse(resolve("research/kaggle_run.py").read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS]
    assert {f.name for f in funcs} == set(HELPERS)
    ns = {"os": os, "Path": Path}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), "kaggle_run_helpers", "exec"), ns)
    return ns


def touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")


def test_finds_code_nested_inside_dataset(tmp_path, helpers):
    """Code zip extracted into a subfolder: REPO_INPUT points ABOVE the code."""
    touch(tmp_path / "veridoc-data" / "veridoc_code" / "research" / "train.py")
    hits = helpers["find_dirs_containing"]("research/train.py", [str(tmp_path / "veridoc-data")])
    assert hits == [tmp_path / "veridoc-data" / "veridoc_code"]


def test_falls_back_to_other_input_roots(tmp_path, helpers):
    """REPO_INPUT points at the data dataset; the code lives in another dataset."""
    touch(tmp_path / "veridoc-data" / "data" / "manifests" / "train.jsonl")
    touch(tmp_path / "veridoc-code" / "research" / "train.py")
    hits = helpers["find_dirs_containing"]("research/train.py", [str(tmp_path / "veridoc-data"), str(tmp_path)])
    assert hits == [tmp_path / "veridoc-code"]


def test_finds_data_root_with_manifests_and_generated(tmp_path, helpers):
    root = tmp_path / "veridoc-data" / "data"
    touch(root / "manifests" / "train.jsonl")
    touch(root / "generated" / "train" / "a.jpg")
    touch(tmp_path / "other" / "manifests" / "train.jsonl")  # decoy without generated/
    hits = [h for h in helpers["find_dirs_containing"]("manifests/train.jsonl", [str(tmp_path)])
            if (h / "generated").is_dir()]
    assert hits == [root]


def test_nothing_found_and_listing(tmp_path, helpers, capsys):
    (tmp_path / "veridoc-data" / "something").mkdir(parents=True)
    assert helpers["find_dirs_containing"]("research/train.py", [str(tmp_path)]) == []
    helpers["show_input_tree"](str(tmp_path))
    out = capsys.readouterr().out
    assert "veridoc-data/" in out and "something/" in out
    helpers["show_input_tree"](str(tmp_path / "missing"))
    assert "does not exist" in capsys.readouterr().out


def test_input_roots(helpers):
    assert helpers["input_roots"]("") == ["/kaggle/input"]
    assert helpers["input_roots"]("/kaggle/input/x") == ["/kaggle/input/x", "/kaggle/input"]


def test_new_kaggle_layout_datasets_owner_slug(tmp_path, helpers):
    """Newer Kaggle mounts: /kaggle/input/datasets/<owner>/<slug>/... (observed on the user's kernel)."""
    inp = tmp_path / "input"
    data = inp / "datasets" / "owner" / "veridoc-data" / "data"
    touch(data / "manifests" / "train.jsonl")
    touch(data / "generated" / "train" / "a.jpg")
    touch(inp / "datasets" / "owner" / "veridoc-code" / "veridoc_code" / "research" / "train.py")
    stale = str(inp / "veridoc-data")  # a REPO_INPUT that does not exist must not break the search
    code = helpers["find_dirs_containing"]("research/train.py", [stale, str(inp)])
    assert code == [inp / "datasets" / "owner" / "veridoc-code" / "veridoc_code"]
    found = [h for h in helpers["find_dirs_containing"]("manifests/train.jsonl", [str(inp)])
             if (h / "generated").is_dir()]
    assert found == [data]
