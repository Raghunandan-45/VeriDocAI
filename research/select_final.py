"""Apply the pre-registered run-1 vs run-2 decision rule (VALIDATION only) and record it.

    .venv/bin/python -m research.select_final

Rule (fixed before run 2 was trained): the final model is the run with the higher
full-validation macro pooled pixel F1 = mean of the splice and text-overwrite pooled F1 at
that run's own frozen pixel threshold (from research/evaluate.py). Ties keep run 1.
Writes results/final_model.json, which names the frozen config Phase 4 must use.
"""
from __future__ import annotations

import json

from research.config import resolve

RUNS = {
    "run1": {"val": "results/val_model.json", "frozen": "results/frozen_config.json",
             "checkpoint": "artifacts/best.pt",
             "description": "baseline: batch Dice, random val subset, pooled-F1 selection, 10 epochs"},
    "run2": {"val": "results/val_model_run2.json", "frozen": "results/frozen_config_run2.json",
             "checkpoint": "artifacts/run2/best.pt",
             "description": "per-crop Dice + stratified selection + 20 epochs"},
}
SPLICE, OVERWRITE = "manipulation_type=splice", "manipulation_type=text_overwrite"


def macro_pooled_f1(val: dict) -> dict:
    g = val["metrics"]["breakdown_tampered_groups"]
    s, o = g[SPLICE]["pixel_pooled"]["f1"], g[OVERWRITE]["pixel_pooled"]["f1"]
    return {"f1_splice": s, "f1_overwrite": o, "macro_pooled_f1": (s + o) / 2}


def main():
    rows = {}
    for name, r in RUNS.items():
        val = json.loads(resolve(r["val"]).read_text())
        frozen = json.loads(resolve(r["frozen"]).read_text())
        if val["split"] != "val" or frozen["selected_on"] != "val":
            raise SystemExit(f"{name}: not a validation result")
        if val["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
            raise SystemExit(f"{name}: val result and frozen config come from different checkpoints")
        rows[name] = {**r, **macro_pooled_f1(val), "checkpoint_sha256": frozen["checkpoint_sha256"],
                      "checkpoint_epoch": frozen["checkpoint_epoch"], "pixel_threshold": frozen["pixel_threshold"]}
    chosen = "run2" if rows["run2"]["macro_pooled_f1"] > rows["run1"]["macro_pooled_f1"] else "run1"
    out = {"evidence_class": "SYNTHETIC validation; rule fixed before run 2; no test data used",
           "rule": "higher full-validation macro pooled F1 (mean of splice and text-overwrite pooled F1 "
                   "at the run's frozen pixel threshold); ties keep run1",
           "runs": rows, "chosen": chosen, "frozen_config_for_phase4": rows[chosen]["frozen"],
           "checkpoint_for_phase4": rows[chosen]["checkpoint"]}
    path = resolve("results/final_model.json")
    path.write_text(json.dumps(out, indent=2))
    for name, r in rows.items():
        print(f"{name}: macro {r['macro_pooled_f1']:.4f} (splice {r['f1_splice']:.4f}, overwrite {r['f1_overwrite']:.4f})")
    print("chosen:", chosen, "->", out["frozen_config_for_phase4"])
    print("wrote", path)


if __name__ == "__main__":
    main()
