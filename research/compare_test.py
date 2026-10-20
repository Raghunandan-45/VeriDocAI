"""Frozen TEST results, clean vs the quick robustness perturbations (synthetic held-out test split).

    .venv/bin/python -m research.compare_test
    # needs results/test_model.json; adds results/test_model_perturb_{jpeg,resize,blur}.json
    # if they exist

Writes results/test_robustness.csv. Every value is copied from those JSON files, except the
"flagged with no correct pixel" rows, which are counted from the per-image CSVs with the
frozen image threshold (flagged = risk score >= threshold, as in docforge.scoring).
Nothing is selected here: thresholds and score rule come from the validation-frozen config.
"""
from __future__ import annotations

import csv
import json

import pandas as pd

from research.config import resolve

SOURCES = [("clean", "results/test_model"), ("jpeg", "results/test_model_perturb_jpeg"),
           ("resize", "results/test_model_perturb_resize"), ("blur", "results/test_model_perturb_blur")]
GROUPS = [("all", None), ("splice", "manipulation_type=splice"), ("text overwrite", "manipulation_type=text_overwrite"),
          ("overwrite easy", "difficulty=easy"), ("overwrite hard", "difficulty=hard"),
          ("splice block", "splice_size_class=block"), ("splice field", "splice_size_class=field")]


def ci(c):
    return None if not c or c.get("lo") is None else f"[{c['lo']:.4f}, {c['hi']:.4f}]"


def group_metrics(m: dict) -> list[tuple[str, float | None, dict | None]]:
    a = m.get("at_frozen_image_threshold") or {}
    return [("pooled pixel F1", m["pixel_pooled"]["f1"], m["pixel_pooled_f1_ci"]),
            ("pooled pixel IoU", m["pixel_pooled"]["iou"], None),
            ("per-image pixel F1 (tampered)", m["pixel_per_image_mean_over_tampered"]["f1"], m["pixel_per_image_f1_ci"]),
            ("image AUC (vs all authentic)", m["image_auc"], m["image_auc_ci"]),
            ("recall at frozen image threshold", a.get("recall"), a.get("recall_ci"))]


def flagged_without_tp(per_image: pd.DataFrame, thr: float) -> dict:
    """Tampered pages flagged at the frozen image threshold although no predicted pixel
    overlaps the ground-truth mask (the risk score came from somewhere else on the page)."""
    t = per_image[per_image["label"] == 1]
    flagged = t[t["score"] >= thr]
    return {"n_flagged": len(flagged), "n_flagged_no_tp": int((flagged["tp"] == 0).sum())}


def main():
    res = {k: json.loads(resolve(p + ".json").read_text()) for k, p in SOURCES if resolve(p + ".json").exists()}
    conds = list(res)
    rows = []
    for gname, key in GROUPS:
        per = {c: group_metrics(r["metrics"]["overall"] if key is None else r["metrics"]["breakdown_tampered_groups"][key])
               for c, r in res.items()}
        for i, (mname, _, _) in enumerate(per[conds[0]]):
            row = {"group": gname, "metric": mname}
            for c in conds:
                val, cc = per[c][i][1], per[c][i][2]
                row[c] = None if val is None else round(val, 4)
                row[f"{c}_ci95"] = ci(cc)
            rows.append(row)
    for mname, fn in [("FPR at frozen image threshold", lambda r: (r["metrics"]["overall"]["at_frozen_image_threshold"]["fpr"],
                                                                   r["metrics"]["overall"]["at_frozen_image_threshold"]["fpr_ci"])),
                      ("authentic pages with any predicted pixel",
                       lambda r: (r["metrics"]["authentic_pages"]["frac_with_any_predicted_pixel"], None)),
                      ("authentic mean predicted area fraction",
                       lambda r: (r["metrics"]["authentic_pages"]["mean_predicted_area_fraction"], None))]:
        row = {"group": "authentic", "metric": mname}
        for c in conds:
            val, cc = fn(res[c])
            row[c], row[f"{c}_ci95"] = round(val, 5), ci(cc)
        rows.append(row)
    thr = res["clean"]["frozen_config"]["image_threshold"]
    for mname in ("n_flagged", "n_flagged_no_tp"):
        row = {"group": "tampered", "metric": {"n_flagged": "tampered pages flagged at frozen image threshold",
                                               "n_flagged_no_tp": "... of which with no correct pixel (tp = 0)"}[mname]}
        for c, p in SOURCES:
            if c in res:
                row[c] = flagged_without_tp(pd.read_csv(resolve(p + "_per_image.csv")), thr)[mname]
        rows.append(row)
    rows.append({"group": "perturbation", "metric": "parameter range used (min-max)",
                 **{c: None if "perturbation" not in r else
                    f"{r['perturbation']['param_min']}-{r['perturbation']['param_max']}" for c, r in res.items()}})
    out = resolve("results/test_robustness.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "metric"] + [k for c in conds for k in (c, f"{c}_ci95")])
        w.writeheader()
        w.writerows(rows)
    print("SYNTHETIC held-out test split, validation-frozen config; perturbations applied post hoc")
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
