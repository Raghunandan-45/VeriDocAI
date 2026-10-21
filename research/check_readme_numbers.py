"""Check every number in the README tables against the results files.

    .venv/bin/python -m research.check_readme_numbers            # check README.md (exit 1 on mismatch)
    .venv/bin/python -m research.check_readme_numbers --render   # print every table from the files
    .venv/bin/python -m research.check_readme_numbers --render test_overall
    .venv/bin/python -m research.check_readme_numbers --fill     # rewrite README tables from the files

How it works:
- Each numeric table in README.md is preceded by a marker `<!-- check:NAME -->`. TABLES below
  holds, for every NAME, the full table as a spec: label cells are plain text, and value cells
  are format templates whose numbers are read from results/*.json / *.csv (or, for settings,
  configs/*.yaml). Templates may not contain literal digits, so every digit in a checked cell
  comes from a file.
- Check: for every marked table, the numbers in each README cell must equal, as written, the
  numbers rendered from the files (same rounding). Digits glued to a letter (F1, p95, run2)
  are names, not numbers. Text inside `backticks` (paths, versions,
  hashes) and markdown emphasis are ignored. The check fails if a README table that contains
  digits has no marker, if a marker has no spec, or if a spec is not used in the README.
- Not checked: numbers in prose. `--prose FILE` lists prose numbers (README or
  docs/INTERVIEW_NOTES.md) that do not appear in any rendered table, for manual review.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import string
import sys
from functools import lru_cache

import yaml

from research.config import resolve

NUM = re.compile(r"(?<![A-Za-z\d.])\d+(?:\.\d+)?")  # digits glued to a letter (F1, p95, run2) are names


# --- value sources -----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _load(path: str):
    p = resolve(path)
    if path.endswith(".json"):
        return json.loads(p.read_text())
    if path.endswith((".yaml", ".yml")):
        return yaml.safe_load(p.read_text())
    if path.endswith(".csv"):
        with open(p, newline="") as f:
            return list(csv.DictReader(f))
    if path.endswith(".txt"):
        return dict(line.split("=", 1) for line in p.read_text().split())
    raise ValueError(path)


class V:
    """A value at a key path in a JSON/YAML/key=value file, times `scale`."""

    def __init__(self, path: str, *keys, scale: float = 1.0):
        self.path, self.keys, self.scale = path, keys, scale

    def get(self):
        node = _load(self.path)
        for k in self.keys:
            node = node[k]
        if isinstance(node, str):
            try:
                node = float(node) if "." in node else int(node)
            except ValueError:
                return node
        return node * self.scale if self.scale != 1.0 else node

    def __repr__(self):
        return f"{self.path}:{'.'.join(map(str, self.keys))}"


class Csv(V):
    """Aggregate of column `col` over CSV rows matching `where` (op: one, sum, count, median)."""

    def __init__(self, path: str, col: str, op: str = "one", scale: float = 1.0, **where):
        super().__init__(path, scale=scale)
        self.col, self.op, self.where = col, op, where

    def get(self):
        rows = [r for r in _load(self.path) if all(r[k] == v for k, v in self.where.items())]
        if self.op == "count":
            return len(rows)
        vals = [float(r[self.col]) for r in rows]
        if self.op == "one":
            if len(vals) != 1:
                raise ValueError(f"{self!r}: {len(vals)} matching rows")
            out = vals[0]
        else:
            out = {"sum": sum, "median": statistics.median}[self.op](vals)
        return out * self.scale

    def __repr__(self):
        return f"{self.path}[{self.where}].{self.col}({self.op})"


class Cell:
    """A value cell: a format template (no literal digits) and the values it formats."""

    def __init__(self, template: str, *values: V):
        bare = re.sub(r"\{[^}]*\}", "", template)
        if NUM.findall(re.sub(r"`[^`]*`", "", bare)):
            raise ValueError(f"literal digits in template {template!r}; take them from a file")
        self.template, self.values = template, values

    def render(self) -> str:
        return self.template.format(*[v.get() for v in self.values])


def c(template: str, *values: V) -> Cell:
    return Cell(template, *values)


def ci3(path: str, *keys) -> tuple[V, V]:
    return V(path, *keys, "lo"), V(path, *keys, "hi")


def f_ci(path: str, val_keys: tuple, ci_keys: tuple, fmt: str = ".3f") -> Cell:
    """'0.537 [0.452, 0.608]'"""
    return c(f"{{0:{fmt}}} [{{1:{fmt}}}, {{2:{fmt}}}]", V(path, *val_keys), *ci3(path, *ci_keys))


# --- table specs ---------------------------------------------------------------------------------
TM, TE = "results/test_model.json", "results/test_ela.json"
VE, V1, V2 = "results/val_ela.json", "results/val_model.json", "results/val_model_run2.json"
VS192, VS256 = "results/val_model_run2_stride192.json", "results/val_model_run2_stride256.json"
LAT, DK = "results/latency_cpu.json", "results/docker_verification.json"
IMP, REG = "results/overwrite_imperfections.json", "results/failure_regions_val.json"
ROB, VCMP, CNT = "results/test_robustness.csv", "results/val_comparison.csv", "results/dataset_counts.csv"
PJ, PR, PB = ("results/test_model_perturb_jpeg.json", "results/test_model_perturb_resize.json",
              "results/test_model_perturb_blur.json")
O = ("metrics", "overall")
G = ("metrics", "breakdown_tampered_groups")


def grp(name):
    return G + (name,)


def pooled(path, keys, ci=True):
    if ci:
        return f_ci(path, keys + ("pixel_pooled", "f1"), keys + ("pixel_pooled_f1_ci",))
    return c("{0:.3f}", V(path, *keys, "pixel_pooled", "f1"))


def auc(path, keys, ci=True):
    if ci:
        return f_ci(path, keys + ("image_auc",), keys + ("image_auc_ci",))
    return c("{0:.3f}", V(path, *keys, "image_auc"))


def dataset_rows():
    rows = [["Split", "Samples", "Authentic", "Splice (block / field)", "Text overwrite (easy / hard)"],
            ["---"] * 5]
    for split in ("train", "val", "test"):
        rows.append([split, c("{0}", V("results/dataset_manifest_hash.txt", f"count_{split}")),
                     c("{0:.0f}", Csv(CNT, "count", "sum", split=split, manipulation_type="none")),
                     c("{0:.0f} / {1:.0f}", Csv(CNT, "count", "sum", split=split, splice_size_class="block"),
                       Csv(CNT, "count", "sum", split=split, splice_size_class="field")),
                     c("{0:.0f} / {1:.0f}", Csv(CNT, "count", "sum", split=split, difficulty="easy"),
                       Csv(CNT, "count", "sum", split=split, difficulty="hard"))])
    rows.append(["**total**", c("{0}", V("results/dataset_manifest_hash.txt", "num_samples")),
                 c("{0:.0f}", Csv(CNT, "count", "sum", manipulation_type="none")),
                 c("{0:.0f} / {1:.0f}", Csv(CNT, "count", "sum", splice_size_class="block"),
                   Csv(CNT, "count", "sum", splice_size_class="field")),
                 c("{0:.0f} / {1:.0f}", Csv(CNT, "count", "sum", difficulty="easy"),
                   Csv(CNT, "count", "sum", difficulty="hard"))])
    return rows


def runs_rows():
    K1, K2 = "results/kaggle_run/metrics.csv", "results/kaggle_run2/metrics.csv"
    FM = "results/final_model.json"
    rows = [["Run", "Training change", "Epochs", "Best epoch", "Median s / epoch (`T4` GPU)",
             "Val macro pooled F1 (decision metric)"], ["---"] * 6]
    for name, k, desc in (("`run1` (baseline)", K1, "batch Dice, random val subset, pooled-F1 selection"),
                          ("**`run2` (final)**", K2, "**per-crop Dice + stratified selection + more epochs**")):
        run = "run1" if k == K1 else "run2"
        rows.append([name, desc, c("{0}", Csv(k, "epoch", "count")),
                     c("{0}", V(FM, "runs", run, "checkpoint_epoch")),
                     c("{0:.0f}", Csv(k, "epoch_seconds", "median")),
                     c("{0:.3f}", V(FM, "runs", run, "macro_pooled_f1"))])
    return rows


def diag_rows():
    DC, DP, DX = ("results/diag_overwrite_crops.json", "results/diag_overwrite_preds.json",
                  "results/diag_overwrite_context.json")
    cat = "detected (some mask pixels >= frozen thr)"
    return [["Finding behind the retrain (no test data used)", "Value", "Source"], ["---"] * 3,
            ["Text overwrites: share of positive training crops",
             c("{0:.0%}", V(DC, "groups", "manipulation_type", "text_overwrite", "share_of_positive_crops")),
             "`diag_overwrite_crops.json`"],
            ["Text overwrites: share of positive training pixels",
             c("{0:.0%}", V(DC, "groups", "manipulation_type", "text_overwrite", "share_of_positive_pixels")),
             "`diag_overwrite_crops.json`"],
            ["Text overwrites: share of tampered pixels in the checkpoint-selection subset",
             c("{0:.1%}", V(DX, "val_subset_used_for_checkpoint_selection", "text_overwrite_share_of_mask_pixels")),
             "`diag_overwrite_context.json`"],
            ["Sampled TRAINING overwrite pages with any detected pixel",
             c("{0} of {1}", V(DP, "summary", "train/all", "categories", cat), V(DP, "summary", "train/all", "n")),
             "`diag_overwrite_preds.json`"],
            ["Median max score inside missed val overwrite masks",
             c("{0:.3f}", V(DP, "summary", "val/all", "median_in_max")), "`diag_overwrite_preds.json`"]]


def val_rows():
    def three(fn):
        return [fn(VE), fn(V1), fn(V2)]
    rows = [["Metric (validation; thresholds selected on these pages)", "ELA", "U-Net `run1`", "U-Net `run2`"],
            ["---"] * 4,
            ["Pixel threshold (selected on val)", *three(lambda p: c("{0:.3f}", V(p, "selection", "pixel_threshold")))],
            ["**Macro pooled F1 (pre-registered decision metric)**",
             c("**{0:.3f}**", Csv(VCMP, "ela", metric="macro pooled F1 (mean of splice and overwrite)")),
             *[c("**{0:.3f}**", V("results/final_model.json", "runs", r, "macro_pooled_f1")) for r in ("run1", "run2")]],
            ["Pooled pixel F1, splice", c("{0:.3f}", V(VE, *grp("manipulation_type=splice"), "pixel_pooled", "f1")),
             pooled(V1, grp("manipulation_type=splice")), pooled(V2, grp("manipulation_type=splice"))],
            ["Pooled pixel F1, text overwrite",
             c("{0:.3f}", V(VE, *grp("manipulation_type=text_overwrite"), "pixel_pooled", "f1")),
             pooled(V1, grp("manipulation_type=text_overwrite")), pooled(V2, grp("manipulation_type=text_overwrite"))],
            ["└ easy", *three(lambda p: c("{0:.3f}", V(p, *grp("difficulty=easy"), "pixel_pooled", "f1")))],
            ["└ hard", *three(lambda p: c("{0:.3f}", V(p, *grp("difficulty=hard"), "pixel_pooled", "f1")))],
            ["Pooled pixel F1, all pages", pooled(VE, O, ci=False), pooled(V1, O), pooled(V2, O)],
            ["Image AUC (selected score rule)", auc(VE, O, ci=False), auc(V1, O), auc(V2, O)],
            [c("FPR at the frozen {0:.0%}-recall image threshold", V(V2, "selection", "target_recall")),
             *three(lambda p: c("{0:.2f}", V(p, *O, "at_frozen_image_threshold", "fpr")))],
            ["Authentic pages with any pixel above the pixel threshold",
             *three(lambda p: c("{0:.2f}", V(p, "metrics", "authentic_pages", "frac_with_any_predicted_pixel")))]]
    return rows


def test_overall_rows():
    def both(fn):
        return [fn(TE), fn(TM)]
    at = O + ("at_frozen_image_threshold",)
    return [["Metric (test split, one frozen run each)", "ELA", "U-Net `run2`"], ["---"] * 3,
            ["Pages (tampered / authentic)", *both(lambda p: c("{0} / {1}", V(p, *O, "n_tampered"), V(p, *O, "n_authentic")))],
            ["Frozen pixel threshold", *both(lambda p: c("{0:.3f}", V(p, "frozen_config", "pixel_threshold")))],
            ["**Pooled pixel F1**", *both(lambda p: pooled(p, O))],
            ["Pooled pixel IoU", *both(lambda p: c("{0:.3f}", V(p, *O, "pixel_pooled", "iou")))],
            ["Per-image pixel F1 (tampered pages)",
             *both(lambda p: f_ci(p, O + ("pixel_per_image_mean_over_tampered", "f1"), O + ("pixel_per_image_f1_ci",)))],
            ["Image AUC", *both(lambda p: auc(p, O))],
            ["Recall at the frozen image threshold", *both(lambda p: f_ci(p, at + ("recall",), at + ("recall_ci",)))],
            ["**FPR at the frozen image threshold**", *both(lambda p: f_ci(p, at + ("fpr",), at + ("fpr_ci",)))],
            ["Authentic pages with any predicted pixel",
             *both(lambda p: c("{0:.2f}", V(p, "metrics", "authentic_pages", "frac_with_any_predicted_pixel")))]]


def test_group_rows():
    rows = [["Tampered group (test)", "n", "U-Net pooled F1", "U-Net per-image F1", "U-Net image AUC",
             "U-Net recall at frozen threshold", "ELA pooled F1"], ["---"] * 7]
    for label, g in (("Splice", "manipulation_type=splice"), ("└ block", "splice_size_class=block"),
                     ("└ field", "splice_size_class=field"), ("Text overwrite", "manipulation_type=text_overwrite"),
                     ("└ easy", "difficulty=easy"), ("└ hard", "difficulty=hard"),
                     ("Invoices (both types)", "document_type=invoice"),
                     ("Marksheets (both types)", "document_type=marksheet")):
        k = grp(g)
        rows.append([label, c("{0}", V(TM, *k, "n_tampered")), pooled(TM, k),
                     c("{0:.3f}", V(TM, *k, "pixel_per_image_mean_over_tampered", "f1")), auc(TM, k),
                     c("{0:.3f}", V(TM, *k, "at_frozen_image_threshold", "recall")),
                     c("{0:.3f}", V(TE, *k, "pixel_pooled", "f1"))])
    return rows


def robust_rows():
    conds = ((TM, "clean"), (PJ, "jpeg"), (PR, "resize"), (PB, "blur"))
    at = O + ("at_frozen_image_threshold",)
    head = ["Metric (test, frozen config)", "Clean",
            c("JPEG re-encode, quality {0}–{1}", V(PJ, "perturbation", "param_min"), V(PJ, "perturbation", "param_max")),
            c("Downscale ×{0:.2f}–{1:.2f}", V(PR, "perturbation", "param_min"), V(PR, "perturbation", "param_max")),
            c("Blur σ {0:.1f}–{1:.1f}", V(PB, "perturbation", "param_min"), V(PB, "perturbation", "param_max"))]
    return [head, ["---"] * 5,
            ["Pooled pixel F1, all", *[pooled(p, O) for p, _ in conds]],
            ["└ splice", *[pooled(p, grp("manipulation_type=splice"), ci=False) for p, _ in conds]],
            ["└ text overwrite", *[pooled(p, grp("manipulation_type=text_overwrite"), ci=False) for p, _ in conds]],
            ["Image AUC", *[auc(p, O) for p, _ in conds]],
            ["Recall / FPR at the frozen image threshold",
             *[c("{0:.3f} / {1:.3f}", V(p, *at, "recall"), V(p, *at, "fpr")) for p, _ in conds]],
            ["Flagged tampered pages with no correct pixel",
             *[c("{0:.0f} of {1:.0f}", Csv(ROB, n, metric="... of which with no correct pixel (tp = 0)"),
                 Csv(ROB, n, metric="tampered pages flagged at frozen image threshold")) for _, n in conds]]]


def region_rows():
    a = ("authentic",)
    rows = [["Region class (authentic val pages)", "Share of false-positive pixels", "Share of page area",
             "Lift (FP share / area share)"], ["---"] * 4]
    for cat, label in (("text", "text"), ("blank_paper", "blank paper"), ("rule_line", "table / rule lines"),
                       ("solid_fill", "solid fill (header bands, shaded cells)")):
        rows.append([label, c("{0:.1%}", V(REG, *a, "fp_pixel_share_by_category", cat)),
                     c("{0:.1%}", V(REG, *a, "page_area_share_by_category", cat)),
                     c("{0:.2f}", V(REG, *a, "fp_rate_lift_vs_area", cat))])
    rows.append(["header zone (`top 15%` of the page)", c("{0:.1%}", V(REG, *a, "fp_pixel_share_top_15pct_of_page")),
                 "", ""])
    return rows


def easy_hard_rows():
    rows = [["Split", "Overwrite difficulty", "n", "Pages with any correct pixel",
             c("Pooled pixel F1 [{0:.0%} CI]", V("configs/eval.yaml", "bootstrap", "level"))],
            ["---"] * 5]
    for split, path in (("val", V2), ("test", TM)):
        for d in ("easy", "hard"):
            rows.append([split, d, c("{0}", V(IMP, split, "by_difficulty", d, "n")),
                         c("{0:.2f}", V(IMP, split, "by_difficulty", d, "detected_frac")),
                         pooled(path, grp(f"difficulty={d}"))])
    return rows


def antialias_rows():
    rows = [["Hard overwrites", "val: with", "val: without", "test: with", "test: without"], ["---"] * 5]
    for imp, label in (("antialias", "changed anti-aliasing"), ("ink_shift", "ink colour shift"),
                       ("letter_spacing", "letter spacing"), ("offset", "small offset")):
        cells = []
        for split in ("val", "test"):
            for w in ("with", "without"):
                k = (IMP, split, "hard_by_imperfection", imp, w)
                cells.append(c("{0:.2f} (n {1})", V(*k, "detected_frac"), V(*k, "n")))
        rows.append([label, *cells])
    return rows


def ela_diag_rows():
    E = "results/ela_diagnostic.json"
    rows = [["ELA post-processing variant (train pages)", "Image AUC (max rule)", "Oracle pooled pixel F1"], ["---"] * 3]
    for v, label in (("standard", "standard pipeline"), ("no_resize_no_blur", "no resize, no blur"),
                     ("single_jpeg_no_resize", "no resize, no blur, single JPEG")):
        rows.append([label, c("{0:.3f}", V(E, "variants", v, "image_auc_max")),
                     c("{0:.3f}", V(E, "variants", v, "oracle_pooled_pixel_f1"))])
    return rows


def latency_rows():
    rows = [["Stride", "Request median / `p95` (s)", "Val pooled pixel F1", "Val splice F1",
             "Val overwrite F1", "Val FPR at frozen threshold", "Authentic val pages with any predicted pixel"],
            ["---"] * 7]
    for s, path in (("128", V2), ("192", VS192), ("256", VS256)):
        r = ("results", s)
        label = c("**{0}** (default; used on test)" if s == "128" else "{0}", V(LAT, *r, "stride"))
        rows.append([label, c("{0:.1f} / {1:.1f}", V(LAT, *r, "request_ms", "p50", scale=1e-3),
                              V(LAT, *r, "request_ms", "p95", scale=1e-3)),
                     pooled(path, O), pooled(path, grp("manipulation_type=splice"), ci=False),
                     pooled(path, grp("manipulation_type=text_overwrite"), ci=False),
                     c("{0:.2f}", V(path, *O, "at_frozen_image_threshold", "fpr")),
                     c("{0:.2f}", V(path, "metrics", "authentic_pages", "frac_with_any_predicted_pixel"))])
    return rows


def latency_setup_rows():
    H, R = ("hardware",), ("resolution",)
    return [["Latency setup", "Value"], ["---"] * 2,
            ["CPU / torch", c("`{0}` / `{1}`", V(LAT, *H, "cpu"), V(LAT, *H, "torch"))],
            ["Logical CPUs / torch threads", c("{0} / {1}", V(LAT, *H, "logical_cpus"), V(LAT, *H, "torch_threads"))],
            ["RAM (GiB)", c("{0}", V(LAT, *H, "ram_gib"))],
            ["Pages timed (validation) / warm-up requests per stride",
             c("{0} / {1}", V(LAT, "n_pages"), V(LAT, "warmup_requests"))],
            ["Page width / height range (px)",
             c("{0}–{1} / {2}–{3}", V(LAT, *R, "width_min"), V(LAT, *R, "width_max"), V(LAT, *R, "height_min"),
               V(LAT, *R, "height_max"))],
            ["Median page size (megapixels)", c("{0}", V(LAT, *R, "median_megapixels"))],
            ["Pages downscaled to the analysis cap", c("{0} (cap {1} px long side)", V(LAT, *R, "n_pages_downscaled_to_cap"),
                                                       V(LAT, *R, "analysis_max_side"))]]


def input_rows():
    S = "configs/serve.yaml"
    st = "results/docker_verification.json"
    return [["Upload check (`serving/validation.py`)", "Limit (`configs/serve.yaml`)", "Response"], ["---"] * 3,
            ["file or request body too large", c("{0:.0f} MiB", V(S, "upload", "max_bytes", scale=1 / 2 ** 20)),
             c("{0}", V(st, "probe", "oversize_11MB", "status"))],
            ["not a JPEG/PNG (judged from the bytes)", "JPEG, PNG",
             c("{0}", V(st, "probe", "non_image", "status"))],
            ["decoded size, read from the header before decoding",
             c("side ≤ {0} px, ≤ {1:.0f} MP, side ≥ {2} px", V(S, "upload", "max_side"),
               V(S, "upload", "max_pixels", scale=1e-6), V(S, "upload", "min_side")), "`422`"],
            ["truncated or corrupt image data", "full decode must succeed", "`400`"],
            ["checkpoint missing or `sha256` ≠ frozen config", "checked at startup", "`503`"]]


def docker_rows():
    P = ("probe",)
    return [["Docker check (`--network none`, checkpoint mounted read-only)", "Result"], ["---"] * 2,
            ["image size (compressed content, MB)", c("{0:.0f}", V(DK, "image_size_bytes", scale=1e-6))],
            ["torch in image / CUDA available", c("`{0}`", V(DK, "torch_in_image"))],
            ["network inside the container", c("{0}", V(DK, "network_inside_container"))],
            ["`GET /health`", c("{0}", V(DK, *P, "health", "http_status"))],
            ["`POST /v1/analyze` (validation splice page)", c("{0}, {1} regions", V(DK, *P, "analyze_page", "status"),
                                                             V(DK, *P, "analyze_page", "n_regions"))],
            ["largest predicted box `[x0, y0, x1, y1]`",
             c("[{0}, {1}, {2}, {3}]", *[V(DK, *P, "analyze_page", "regions", 0, "bbox_xyxy", i) for i in range(4)])],
            ["non-image upload", c("{0}", V(DK, *P, "non_image", "status"))],
            ["oversized upload (announced body over the limit)", c("{0}", V(DK, *P, "oversize_11MB", "status"))]]


TABLES = {
    "dataset": dataset_rows, "runs": runs_rows, "diag": diag_rows, "val_models": val_rows,
    "ela_diag": ela_diag_rows, "test_overall": test_overall_rows, "test_groups": test_group_rows,
    "robustness": robust_rows, "fp_regions": region_rows, "easy_hard": easy_hard_rows,
    "imperfections": antialias_rows, "latency": latency_rows, "latency_setup": latency_setup_rows,
    "inputs": input_rows, "docker": docker_rows,
}


# --- rendering and checking ----------------------------------------------------------------------
def render_cell(cell) -> str:
    if isinstance(cell, Cell):
        return cell.render()
    if numbers(cell):
        raise ValueError(f"label cell {cell!r} contains digits; take them from a file or put them in backticks")
    return cell


def render_table(name: str) -> list[list[str]]:
    return [[render_cell(x) for x in row] for row in TABLES[name]()]


def to_markdown(rows: list[list[str]]) -> str:
    return "\n".join("| " + " | ".join(r) + " |" if r[0] != "---" else "|" + "---|" * len(r) for r in rows)


def numbers(text: str) -> list[str]:
    text = re.sub(r"`[^`]*`", "", text)
    return NUM.findall(text)


def parse_tables(md: str) -> list[tuple[str | None, int, list[list[str]]]]:
    """(marker name or None, line number, rows of cells) for every table in the markdown."""
    lines, out, i = md.splitlines(), [], 0
    in_code = False
    while i < len(lines):
        if lines[i].startswith("```"):
            in_code = not in_code
        if not in_code and lines[i].startswith("|"):
            start = i
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([x.strip() for x in lines[i].strip().strip("|").split("|")])
                i += 1
            j = start - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            m = re.match(r"<!--\s*check:(\w+)\s*-->", lines[j].strip()) if j >= 0 else None
            out.append((m.group(1) if m else None, start + 1, rows))
            continue
        i += 1
    return out


def check(path: str) -> list[str]:
    md = resolve(path).read_text()
    errors, used = [], set()
    for name, line, rows in parse_tables(md):
        has_digits = any(numbers(cell) for r in rows for cell in r if not set(cell) <= set("-: "))
        if name is None:
            if has_digits:
                errors.append(f"{path}:{line}: table with numbers has no <!-- check:NAME --> marker")
            continue
        if name not in TABLES:
            errors.append(f"{path}:{line}: unknown table spec {name!r}")
            continue
        used.add(name)
        exp = render_table(name)
        if len(exp) != len(rows):
            errors.append(f"{path}:{line} [{name}]: {len(rows)} rows, spec has {len(exp)}")
            continue
        for ri, (er, ar) in enumerate(zip(exp, rows)):
            if len(er) != len(ar):
                errors.append(f"{path}:{line + ri} [{name}]: {len(ar)} cells, spec has {len(er)}")
                continue
            for ci, (ec, ac) in enumerate(zip(er, ar)):
                if numbers(ec) != numbers(ac):
                    errors.append(f"{path}:{line + ri} [{name}] cell {ci + 1}: README {numbers(ac)} "
                                  f"!= files {numbers(ec)}  (expected cell: {ec!r})")
    for name in sorted(set(TABLES) - used):
        errors.append(f"{path}: spec {name!r} is not used (no <!-- check:{name} --> table)")
    return errors


def prose_numbers(path: str) -> list[tuple[int, str]]:
    """Decimal or percent numbers in prose (outside tables and code) that appear in no rendered table."""
    known = {n for name in TABLES for r in render_table(name) for cell in r for n in numbers(cell)}
    out, in_code = [], False
    for k, line in enumerate(resolve(path).read_text().splitlines(), 1):
        if line.startswith("```"):
            in_code = not in_code
        if in_code or line.startswith("|") or line.startswith("<!--"):
            continue
        for n in re.findall(r"\d+\.\d+|\d+(?=%)", re.sub(r"`[^`]*`", "", line)):
            if n not in known:
                out.append((k, n))
    return out


def fill(path: str) -> int:
    """Rewrite every marked table (or a `TABLE` placeholder line after a marker) from the files."""
    lines, out, i, n = resolve(path).read_text().splitlines(), [], 0, 0
    while i < len(lines):
        out.append(lines[i])
        m = re.match(r"<!--\s*check:(\w+)\s*-->", lines[i].strip())
        i += 1
        if m:
            while i < len(lines) and (lines[i].startswith("|") or lines[i].strip() == "TABLE"):
                i += 1
            out.extend(to_markdown(render_table(m.group(1))).splitlines())
            n += 1
    resolve(path).write_text("\n".join(out) + "\n")
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--render", nargs="*", default=None, help="print tables rendered from the files")
    ap.add_argument("--prose", default=None, help="list prose numbers not found in any table")
    ap.add_argument("--fill", action="store_true", help="rewrite the README tables from the files")
    a = ap.parse_args(argv)
    if a.fill:
        print(f"filled {fill(a.readme)} tables in {a.readme}")
        return 0
    if a.render is not None:
        for name in a.render or TABLES:
            print(f"<!-- check:{name} -->\n{to_markdown(render_table(name))}\n")
        return 0
    if a.prose:
        for k, n in prose_numbers(a.prose):
            print(f"{a.prose}:{k}: {n}")
        return 0
    errors = check(a.readme)
    for e in errors:
        print("MISMATCH", e)
    n_tables = sum(1 for name, _, _ in parse_tables(resolve(a.readme).read_text()) if name)
    print(f"{a.readme}: {n_tables} checked tables, {len(errors)} problem(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
