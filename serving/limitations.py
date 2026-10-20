"""The fixed limitations text returned with every analysis.

The numbers are copied from the single frozen test evaluation (synthetic held-out test
split, stride 128). tests/test_serving.py checks each one against the results files, so the
text cannot silently drift from the measured values:
- results/test_model.json: splice / text-overwrite pixel F1, image AUC, recall and FPR at
  the validation-frozen image threshold;
- results/test_model_perturb_blur.json: pixel F1 after Gaussian blur (sigma 1.2-1.6).
"""

LIMITATIONS = (
    "Experimental research prototype. "
    "The model was evaluated on synthetic data only (computer-generated fictional invoices and "
    "marksheets); its accuracy on real documents has not been measured. "
    "It localizes pasted regions (splices) much better than edited text: on the synthetic "
    "held-out test split, pixel F1 was 0.62 for splices and 0.19 for text overwrites "
    "(results/test_model.json). "
    "It is sensitive to blur: after Gaussian blur (sigma 1.2-1.6) pixel F1 fell from 0.54 to "
    "0.33 overall and from 0.19 to 0.02 for text overwrites "
    "(results/test_model_perturb_blur.json). "
    "The image-level risk score is experimental and not reliable as a screening filter: at the "
    "threshold chosen on validation for 90% recall, it flagged 94% of tampered and 92% of "
    "authentic test pages (image AUC 0.70, results/test_model.json). "
    "This tool gives no tampered/authentic verdict. Highlighted regions are places to inspect, "
    "and the absence of highlights does not show that a document is unedited."
)

# (value as written above, results file, JSON path) for the consistency test
CITED_NUMBERS = [
    ("0.62", "results/test_model.json", ["metrics", "breakdown_tampered_groups", "manipulation_type=splice", "pixel_pooled", "f1"]),
    ("0.19", "results/test_model.json", ["metrics", "breakdown_tampered_groups", "manipulation_type=text_overwrite", "pixel_pooled", "f1"]),
    ("0.54", "results/test_model.json", ["metrics", "overall", "pixel_pooled", "f1"]),
    ("0.33", "results/test_model_perturb_blur.json", ["metrics", "overall", "pixel_pooled", "f1"]),
    ("0.02", "results/test_model_perturb_blur.json", ["metrics", "breakdown_tampered_groups", "manipulation_type=text_overwrite", "pixel_pooled", "f1"]),
    ("94%", "results/test_model.json", ["metrics", "overall", "at_frozen_image_threshold", "recall"]),
    ("92%", "results/test_model.json", ["metrics", "overall", "at_frozen_image_threshold", "fpr"]),
    ("0.70", "results/test_model.json", ["metrics", "overall", "image_auc"]),
]
