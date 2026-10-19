"""ELA vs U-Net on VALIDATION (synthetic; thresholds selected on the same data -> optimistic).

    .venv/bin/python -m research.compare_val      # needs results/val_ela.json and results/val_model.json

Writes results/val_comparison.csv; every value is copied from those two JSON files.
"""
from __future__ import annotations

import csv
import json

from research.config import resolve


def row(name, ela, model, key_fn):
    out = {"metric": name}
    for label, res in (("ela", ela), ("model", model)):
        val, ci = key_fn(res)
        out[label] = None if val is None else round(val, 4)
        out[f"{label}_ci95"] = None if not ci or ci.get("lo") is None else f"[{ci['lo']:.4f}, {ci['hi']:.4f}]"
    return out


def main():
    ela = json.loads(resolve("results/val_ela.json").read_text())
    model = json.loads(resolve("results/val_model.json").read_text())
    o = lambda r: r["metrics"]["overall"]  # noqa: E731
    rows = [
        row("pixel_threshold (val-selected)", ela, model, lambda r: (r["selection"]["pixel_threshold"], None)),
        row("pooled pixel F1", ela, model, lambda r: (o(r)["pixel_pooled"]["f1"], o(r)["pixel_pooled_f1_ci"])),
        row("pooled pixel IoU", ela, model, lambda r: (o(r)["pixel_pooled"]["iou"], None)),
        row("per-image pixel F1 (tampered)", ela, model,
            lambda r: (o(r)["pixel_per_image_mean_over_tampered"]["f1"], o(r)["pixel_per_image_f1_ci"])),
        row("image AUC (val-selected rule)", ela, model, lambda r: (o(r)["image_auc"], o(r)["image_auc_ci"])),
        row("val recall at frozen image threshold", ela, model,
            lambda r: ((o(r).get("at_frozen_image_threshold") or {}).get("recall"), None)),
        row("val FPR at frozen image threshold", ela, model,
            lambda r: ((o(r).get("at_frozen_image_threshold") or {}).get("fpr"),
                       (o(r).get("at_frozen_image_threshold") or {}).get("fpr_ci"))),
    ]
    rows.append({"metric": "score rule", "ela": ela["selection"]["score_rule"], "model": model["selection"]["score_rule"]})
    out = resolve("results/val_comparison.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "ela", "ela_ci95", "model", "model_ci95"])
        w.writeheader()
        w.writerows(rows)
    print("SYNTHETIC validation comparison (thresholds selected on validation; optimistic)")
    for r in rows:
        print(r)
    print("wrote", out)


if __name__ == "__main__":
    main()
