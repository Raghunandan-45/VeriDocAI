# Failure analysis (Phase 4)

**Synthetic evaluation only.** The pages are fictional, Pillow-rendered invoices and
marksheets. Nothing here is evidence of real-world performance. The model is run 2
(`artifacts/run2/best.pt`) with the validation-frozen config
`results/frozen_config_run2.json`:
- pixel threshold 0.88;
- image score = mean of the top-500 pixel scores;
- image threshold 0.889 (90% recall on validation).

No threshold, rule or model was changed after the single test evaluation.

**Every explanation below is a hypothesis.** Each one is drawn from a handful of pages or
from small groups (n = 7 to 30), without controlled experiments, and none was tested by
changing the data or the model.

Sources:

| File | Content | Split |
|---|---|---|
| `results/failure_examples.json` | the 12 examples below | test |
| `docs/figures/failure_examples.jpg` | all 12 panels in one figure (`results/failures/NN_*.png` are the separate panels, gitignored) | test |
| `results/overwrite_imperfections.json` | text overwrites by imperfection, easy mode, field and mask size | val and test |
| `results/failure_regions_val.json` | where false-positive pixels fall on authentic pages; which part of an overwrite mask is found | val only |
| `results/test_model.json`, `results/test_robustness.csv` | headline test numbers | test |

## How the examples were chosen

Picks are deterministic, made by `research/failure_analysis.py examples` from
`results/test_model_per_image.csv` before the panels were looked at:

| Pick | Rule | Count |
|---|---|---|
| Missed hard overwrites | per-image F1 = 0, largest masks | 2 |
| Missed easy overwrites | per-image F1 = 0, largest masks | 2 |
| Best-localised overwrites | highest per-image F1 | 2 |
| Worst-localised splices | lowest per-image F1 | 2 |
| Best-localised splice | highest per-image F1 | 1 |
| Authentic pages with the highest risk score | | 2 |
| Tampered page with the lowest risk score | | 1 |

Panel legend:
- **Left:** the page. The red box is the crop shown; magenta marks predicted pixels.
- **Middle:** the crop, with the ground-truth mask in green.
- **Right:** the score heatmap, with ground truth (green) and prediction (magenta).

## The 12 annotated examples

"Score in mask" is the highest pixel score inside the ground-truth mask; "risk" is the image
risk score; "flagged" means risk ≥ 0.889.

### 1. `invoice_0048_text_overwrite_0`: missed hard overwrite
- **Edit:** date "February 28, 2024" → "April 06, 2022", same font (Liberation Mono 18).
  Imperfections: offset (−1.7, +1.0) px and an ink-colour shift.
- **Result:** F1 0.00; score in mask 0.027; risk 0.988, flagged.
- **What the panel shows:** the date gets no response. The prediction lands on the invoice
  number "LQV/1194/25" just above it, and there is a weaker response on "December".
- **Hypothesis:** an edit in the same font with a sub-pixel offset and slight ink shift is
  not visible to the model. **The page is flagged for the wrong reason**: the risk score
  comes from a false positive on a nearby authentic field.

### 2. `invoice_0124_text_overwrite_0`: missed hard overwrite
- **Edit:** the bold grand total "Rs. 3,515,612.73" → "Rs. 11,386,328.82". Same font (DejaVu
  Sans Mono), 24 → 23 px. Imperfection: antialias (4× supersampled rendering).
- **Result:** F1 0.00; score in mask 0.067; risk 0.964, flagged.
- **What the panel shows:** the heat sits on untouched unit-price and amount cells in the
  table above.
- **Hypothesis:** the model responds to digit strings in table cells in general, not to this
  edit. Again the page is flagged for the wrong reason.

### 3. `marksheet_0179_text_overwrite_0`: missed easy overwrite (size)
- **Edit:** guardian name "Nisha Mathew" → "Daniel Fernandes", DejaVu Serif 21 → 25 px. The
  larger size is visible by eye.
- **Result:** F1 0.00; score in mask 0.116; risk 0.893, flagged just above the threshold.
- **What the panel shows:** weak responses on the roll-number digits and the
  "Obtained" column, none on the name.
- **Hypothesis:** the model appears to have learnt local texture cues more than layout
  consistency, such as one line's size against its neighbours. A cleanly rendered edit that
  is only larger has little local texture anomaly. Names were never detected on test
  (0 of 5, `by_field_kind.name`).

### 4. `invoice_0027_text_overwrite_0`: missed easy overwrite (font family)
- **Edit:** due date in Liberation Mono → "August 07, 2022" in DejaVu Sans (proportional),
  same size. To a human it is visibly the odd one out among monospaced values.
- **Result:** F1 0.00; score in mask 0.562, below the 0.88 pixel threshold; risk 0.865, not
  flagged.
- **What the panel shows:** the strongest response is on the neighbouring PO number
  "PO-249600".
- **Hypothesis:** a change of font family alone is the hardest easy mode. Pooled F1 by easy
  mode (`easy_by_mode`):
  - family-only: 0.13 on val (n 7), 0.11 on test (n 9);
  - size-only: 0.41 on val, 0.33 on test.

  Dates were never detected on test (0 of 6).

### 5. `invoice_0138_text_overwrite_0`: best-localised overwrite
- **Edit:** discount "Rs. 15,963.80" → "Rs. 28,726.31". Liberation Serif 21 → Liberation Mono
  16, so both family and size change.
- **Result:** F1 0.80; score in mask 0.998; risk 0.997.
- **What the panel shows:** the edited value is predicted almost exactly. The page also has
  false positives on two untouched amounts ("Rs. 236,773.45", "Rs. 9,596.64").
- **Hypothesis:** the largest kind of change (font family and size, monospace inside serif
  text) is the case the model handles. The false positives on other amounts look like
  example 2.

### 6. `marksheet_0078_text_overwrite_0`: best-localised overwrite
- **Edit:** marks "41" → "43", Liberation Serif 18 → 15 px. The mask is only 357 px.
- **Result:** F1 0.72; score in mask 0.980; risk 0.923.
- **What the panel shows:** the edit is found. There are also weaker responses on the total
  "245" and a grade "A".
- **Hypothesis:** a tiny, clearly smaller numeral in a column of same-size numerals is
  found. Very small edits are therefore not missed automatically. Test detection by field
  (`by_field_kind`, pages with any correct pixel):
  - grades: 5 of 8;
  - integers: 8 of 21.

### 7. `invoice_0022_splice_0`: worst-localised splice (field size)
- **Edit:** a small patch from `invoice_0149` containing "35" (726 px), pasted onto blank
  paper under the footer text.
- **Result:** F1 0.00; score in mask 0.005; risk 0.753, not flagged.
- **What the panel shows:** the response is on the bank account number, not on the pasted
  "35".
- **Hypothesis:** the donor's paper tone and texture closely match the target's, so the
  patch boundary is invisible, and the patch content is ordinary digits. With nothing to
  separate it from the page, the model scores it like normal text.

### 8. `invoice_0066_splice_0`: worst-localised splice (block size)
- **Edit:** a 60,196 px block from `invoice_0101` (payment terms, bank details, footer line),
  pasted onto blank paper.
- **Result:** F1 0.00; score in mask 0.685, below the pixel threshold; risk 0.936, flagged.
- **What the panel shows:** inside the block, only the account-number digits get a
  noticeable response.
- **Hypothesis:** this is a second failure in which the patch background matches the page,
  so the edges carry no cue. Block splices are usually found (block pooled F1 0.60 on
  test), so this failure is probably about background match, not size. The same source's
  authentic page is example 11.

### 9. `marksheet_0128_splice_0`: best-localised splice
- **Edit:** a block from `marksheet_0087`, pasted over the left of the marks table. It cuts
  through the coloured header row and the zebra rows.
- **Result:** F1 0.95; score in mask 0.997; risk 0.997; no false-positive pixels.
- **What the panel shows:** the predicted region follows the patch footprint closely.
- **Hypothesis:** the patch breaks table structure (header colour, row stripes, column
  alignment), and that is the cue the model uses best. Compare examples 7 and 8, whose
  patches sit on blank paper.

### 10. `invoice_0168_authentic_0`: authentic page with the highest risk
- **Result:** risk 0.997, flagged; 6,533 false-positive pixels.
- **What the panel shows:** false positives on untouched table cells: unit price
  "Rs. 11,310.03", quantity "48" and part of an amount. This page was blurred in generation
  (σ 0.99).
- **Hypothesis:** numeric table cells look like the edited values the model saw in training.
  Most overwrites target amounts and integers. The validation region analysis below finds
  the same: false positives are concentrated on text.

### 11. `invoice_0066_authentic_0`: authentic page with the second-highest risk
- **Result:** risk 0.995; 22,924 false-positive pixels.
- **What the panel shows:** a confident prediction on the untouched footer sentence "…with
  your payment.", plus smaller blobs on the signature line and a table cell.
- **Hypothesis:** this is ordinary text with no apparent anomaly. It suggests the model has
  some generator-specific prior about where text appears or what it looks like. It is the
  same source as example 8: the spliced copy is missed, the authentic copy is flagged.

### 12. `invoice_0164_text_overwrite_0`: tampered page with the lowest risk
- **Edit:** invoice date "18.12.2025" → "26.10.2024", Liberation Serif 22 → 17 px (easy, size).
  The new date is visibly smaller.
- **Result:** F1 0.00; score in mask 0.0005; risk 0.667, not flagged.
- **What the panel shows:** no response at all on the date.
- **Hypothesis:** like example 3, a clean size change is not a local texture anomaly, and
  this page also has few false positives elsewhere. So its risk is the lowest of all
  tampered pages.

### Patterns across the examples (hypotheses)

- **Flagged for the wrong reason.** At the frozen image threshold, 113 of 120 tampered test
  pages are flagged, and **36 of those 113 have no correct pixel at all**
  (`results/test_robustness.csv`, rows "tampered pages flagged…"). Their risk score comes
  from false positives elsewhere on the page. Image recall therefore overstates
  localisation.
- **Numeric text attracts false positives.** This holds for authentic pages and for the
  untouched parts of tampered pages (examples 1, 2, 5, 10).
- **Visible layout changes are missed.** A size or font-family change without a local
  texture anomaly is often missed (examples 3, 4, 12). This hypothesis is about what the
  model learnt, not about how hard the edit is to see.
- **Splices on blank paper are missed; splices that cross table structure are found**
  (examples 7 and 8 against 9).

## Question 1: why did hard overwrites score higher than easy ones on validation?

**Short answer: the validation gap was within noise, and it did not repeat on test.**

| Split | Group | n | Pages with a correct pixel | Mean per-image F1 | Pooled F1 [95% CI] |
|---|---|---|---|---|---|
| val | easy | 30 | 0.40 | 0.208 | 0.236 [0.117, 0.383] |
| val | hard | 30 | 0.43 | 0.206 | 0.276 [0.135, 0.417] |
| test | easy | 30 | 0.47 | 0.195 | 0.274 [0.119, 0.416] |
| test | hard | 30 | 0.30 | 0.127 | 0.099 [0.046, 0.179] |

Sources:
- n, pages with a correct pixel, and mean per-image F1: `results/overwrite_imperfections.json`;
- val CIs: `results/val_model_run2.json`;
- test CIs: `results/test_model.json`.

Reading the table:
- On validation, easy and hard are almost identical by fraction detected and by per-image F1.
  The pooled (pixel-weighted) F1 differs, but the CIs overlap almost completely.
- On test the order reverses.
- So there is no evidence that hard overwrites are easier.

**Is the model detecting generator imperfections?** Hard edits on the same page can have
several imperfections, so the "with" and "without" groups overlap and are confounded. The
groups are small, and no CIs were computed. Fraction of hard pages with at least one
correct pixel (`hard_by_imperfection`):

| Imperfection | val with | val without | test with | test without | Same direction on both splits? |
|---|---|---|---|---|---|
| antialias (supersampled or aliased rendering) | 0.58 (n 19) | 0.18 (n 11) | 0.40 (n 15) | 0.20 (n 15) | yes, also by pooled F1 (val 0.34 vs 0.16, test 0.15 vs 0.05) |
| ink_shift | 0.52 (n 21) | 0.22 (n 9) | 0.35 (n 17) | 0.23 (n 13) | by fraction detected only; pooled F1 has no gap (val 0.27 vs 0.28, test 0.10 vs 0.11) |
| letter_spacing | 0.29 (n 17) | 0.62 (n 13) | 0.33 (n 12) | 0.28 (n 18) | no |
| offset | 0.47 (n 17) | 0.38 (n 13) | 0.24 (n 17) | 0.38 (n 13) | no |

**Hypothesis.** The one consistent signal is that hard edits rendered with a different
anti-aliasing are found about twice as often as those without. The model may partly key on
this rendering artifact of our generator rather than on the edit itself. If so, it would
not carry over to real edits made with different software.

Two further hints from validation (`results/failure_regions_val.json`,
`text_overwrite_mask_parts`):
- Median mean score on the new-text ink: hard 0.48, easy 0.25.
- Pixel recall at the threshold is similar for both (0.29 vs 0.31).

**This is not tested.** Testing it would need a controlled re-render of the same edits with
and without each imperfection. That is not done here, because no further test-time or
model changes are allowed in this phase.

Other observations (same file, small n):
- Larger overwrite masks have higher pooled F1 on both splits.
- Font-family-only edits are the weakest easy mode on both splits (see example 4).

## Question 2: where do the false positives on authentic validation pages occur?

At the frozen pixel threshold, 53 of 60 authentic validation pages have at least one
false-positive pixel. That is the "88%" in `results/val_model_run2.json`. On test the figure
is 60 of 60 (`results/test_model.json`, `authentic_pages`).

`research/failure_analysis.py regions` (validation only) splits each page into structure
classes. It then compares each class's share of false-positive pixels with its share of
page area:

| Region class | Share of FP pixels | Share of page area | Lift (FP share / area share) |
|---|---|---|---|
| text | 0.862 | 0.088 | **9.8** |
| blank paper | 0.116 | 0.808 | 0.14 |
| rule lines (table and separator lines) | 0.016 | 0.057 | 0.28 |
| solid fill (header bands, shaded cells) | 0.006 | 0.048 | 0.13 |

More counts from the same analysis:
- **Connected false-positive components:** 189 mostly on text, 19 mostly on blank paper,
  4 mostly on rule lines.
- **Page position:** 10.8% of false-positive pixels lie in the top 15% of the page, where
  the headers are, and 6.1% lie within 32 px of the border.
- **Header style** (total false-positive pixels / pages): band 26,156 / 10; plain
  20,140 / 11; line 10,878 / 9; marksheets (no header-style field) 38,858 / 30. Pages with a
  coloured header band have the most false positives per page, but there are only about 10
  pages per style.

**Answer:** the false positives occur in **ordinary text**, not in headers or table lines.
Text covers 9% of the page but holds 86% of the false-positive pixels. The examples
(10, 11, and the untouched cells in 1, 2 and 5) suggest that numeric table cells and
values are frequent locations. That part was observed in the panels, not counted by
category.

**Hypothesis:** most overwrites are on amounts and integers: 35 of 60 on val and 37 of 60
on test (`by_field_kind`). Training taught the model that "a numeric value in a cell" is where
edits are, and that pattern also fires on untouched values.

Limitations of the region method:
- The classes come from a simple pixel-colour rule (see `structure_method` in the JSON).
- Text drawn inside a header band is counted as solid fill.
- Large bold titles are partly counted as solid fill.

Visually checked on two pages only (`invoice_0026` with a header band, `marksheet_0001`).

**Missed parts of overwrites.** Splitting validation overwrite masks into parts
(`text_overwrite_mask_parts.all`, pixel recall at the threshold):

| Part of the mask | Recall |
|---|---|
| new-text ink | 0.30 |
| background inside the new text box | 0.30 |
| area only under the cover rectangle (old value removed, nothing redrawn) | 0.10 |

So the model mainly responds near the new text. Cover-only pixels are 32% of overwrite
mask pixels, which by itself caps overwrite recall well below 1.

## Robustness check (test split, frozen config, post-hoc perturbations)

Each perturbation is applied alone to the saved test images (`configs/robustness.yaml`, all
values initial assumptions outside the training ranges). Source:
`results/test_robustness.csv`.

| Metric | Clean | JPEG q40–60 | Resize ×0.60–0.75 | Blur σ 1.2–1.6 |
|---|---|---|---|---|
| Pooled pixel F1, all [95% CI] | 0.537 [0.452, 0.608] | 0.507 [0.426, 0.585] | 0.394 [0.296, 0.478] | 0.330 [0.248, 0.406] |
| Pooled pixel F1, splice | 0.618 | 0.581 | 0.567 | 0.499 |
| Pooled pixel F1, text overwrite | 0.187 | 0.162 | 0.077 | 0.022 |
| Image AUC [95% CI] | 0.696 [0.621, 0.773] | 0.658 [0.579, 0.739] | 0.653 [0.577, 0.732] | 0.629 [0.543, 0.711] |
| Recall at frozen image threshold | 0.942 | 0.908 | 0.933 | 0.983 |
| FPR at frozen image threshold | 0.917 | 0.883 | 0.933 | 0.950 |

**Hypothesis:**
- Splice localisation degrades gradually.
- Overwrite localisation collapses under blur, consistent with the model relying on fine
  rendering detail in the new text (see question 1).
- Blur raises recall only because it raises the risk score of almost every page: the FPR
  rises too, to 0.95.
