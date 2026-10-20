"""ELA vs U-Net runs on VALIDATION (synthetic; thresholds selected on the same data -> optimistic).

    .venv/bin/python -m research.compare_val
    # needs results/val_ela.json and results/val_model.json (run 1); adds run 2 if
    # results/val_model_run2.json exists

Writes results/val_comparison.csv; every value is copied from those JSON files (macro F1 is
the mean of the splice and text-overwrite pooled F1, as in research/select_final.py).
"""
from __future__ import annotations

import csv
import json

from research.config import resolve

SOURCES = [("ela", "results/val_ela.json"), ("model_run1", "results/val_model.json"),
           ("model_run2", "results/val_model_run2.json")]


def o(r):
    return r["metrics"]["overall"]


def grp(name):
    return lambda r: r["metrics"]["breakdown_tampered_groups"][f"{name}"]


def ci(c):
    return None if not c or c.get("lo") is None else f"[{c['lo']:.4f}, {c['hi']:.4f}]"


def at_thr(r):
    return o(r).get("at_frozen_image_threshold") or {}


METRICS = [
    ("pixel_threshold (val-selected)", lambda r: (r["selection"]["pixel_threshold"], None)),
    ("pooled pixel F1", lambda r: (o(r)["pixel_pooled"]["f1"], o(r)["pixel_pooled_f1_ci"])),
    ("pooled pixel IoU", lambda r: (o(r)["pixel_pooled"]["iou"], None)),
    ("per-image pixel F1 (tampered)", lambda r: (o(r)["pixel_per_image_mean_over_tampered"]["f1"],
                                                 o(r)["pixel_per_image_f1_ci"])),
    ("pooled pixel F1: splice", lambda r: (grp("manipulation_type=splice")(r)["pixel_pooled"]["f1"],
                                          grp("manipulation_type=splice")(r)["pixel_pooled_f1_ci"])),
    ("pooled pixel F1: text overwrite", lambda r: (grp("manipulation_type=text_overwrite")(r)["pixel_pooled"]["f1"],
                                                  grp("manipulation_type=text_overwrite")(r)["pixel_pooled_f1_ci"])),
    ("pooled pixel F1: overwrite easy", lambda r: (grp("difficulty=easy")(r)["pixel_pooled"]["f1"],
                                                  grp("difficulty=easy")(r)["pixel_pooled_f1_ci"])),
    ("pooled pixel F1: overwrite hard", lambda r: (grp("difficulty=hard")(r)["pixel_pooled"]["f1"],
                                                  grp("difficulty=hard")(r)["pixel_pooled_f1_ci"])),
    ("macro pooled F1 (mean of splice and overwrite)",
     lambda r: ((grp("manipulation_type=splice")(r)["pixel_pooled"]["f1"]
                 + grp("manipulation_type=text_overwrite")(r)["pixel_pooled"]["f1"]) / 2, None)),
    ("image AUC (val-selected rule)", lambda r: (o(r)["image_auc"], o(r)["image_auc_ci"])),
    ("image AUC: splice vs authentic", lambda r: (grp("manipulation_type=splice")(r)["image_auc"],
                                                 grp("manipulation_type=splice")(r)["image_auc_ci"])),
    ("image AUC: overwrite vs authentic", lambda r: (grp("manipulation_type=text_overwrite")(r)["image_auc"],
                                                    grp("manipulation_type=text_overwrite")(r)["image_auc_ci"])),
    ("val recall at frozen image threshold", lambda r: (at_thr(r).get("recall"), None)),
    ("val FPR at frozen image threshold", lambda r: (at_thr(r).get("fpr"), at_thr(r).get("fpr_ci"))),
    ("authentic pages with any predicted pixel", lambda r: (r["metrics"]["authentic_pages"]["frac_with_any_predicted_pixel"], None)),
]


def main():
    res = {label: json.loads(resolve(p).read_text()) for label, p in SOURCES if resolve(p).exists()}
    labels = list(res)
    rows = []
    for name, fn in METRICS:
        row = {"metric": name}
        for label in labels:
            val, c = fn(res[label])
            row[label] = None if val is None else round(val, 4)
            row[f"{label}_ci95"] = ci(c)
        rows.append(row)
    rows.append({"metric": "score rule", **{label: res[label]["selection"]["score_rule"] for label in labels}})
    rows.append({"metric": "checkpoint epoch", **{label: res[label].get("checkpoint_epoch") for label in labels}})
    out = resolve("results/val_comparison.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric"] + [k for label in labels for k in (label, f"{label}_ci95")])
        w.writeheader()
        w.writerows(rows)
    print("SYNTHETIC validation comparison (thresholds selected on validation; optimistic)")
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
