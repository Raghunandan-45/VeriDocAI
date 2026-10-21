"""The README tables match the results files, and the checker catches a changed number."""
from __future__ import annotations

import re

from research import check_readme_numbers as crn
from research.config import resolve


def test_readme_tables_match_results_files():
    assert crn.check("README.md") == []


def test_checker_detects_a_changed_number(tmp_path, monkeypatch):
    md = resolve("README.md").read_text()
    # bump the first three-decimal number inside the test_overall table by 0.001
    start = md.index("<!-- check:test_overall -->")
    m = re.compile(r"\d\.\d{3}").search(md, start)
    old = m.group(0)
    new = f"{float(old) + 0.001:.3f}"
    bad = tmp_path / "README.md"
    bad.write_text(md[: m.start()] + new + md[m.end():])
    errors = crn.check(str(bad))
    assert len(errors) == 1 and "test_overall" in errors[0] and old in errors[0]


def test_unmarked_numeric_table_is_reported(tmp_path):
    bad = tmp_path / "README.md"
    bad.write_text(resolve("README.md").read_text() + "\n| a | b |\n|---|---|\n| x | 0.5 |\n")
    assert any("no <!-- check:NAME --> marker" in e for e in crn.check(str(bad)))


def test_label_cells_cannot_hide_unchecked_digits():
    import pytest
    with pytest.raises(ValueError):
        crn.render_cell("FPR at 90% recall")
    with pytest.raises(ValueError):
        crn.c("quality 40–{0}", crn.V("configs/eval.yaml", "bootstrap", "n"))
