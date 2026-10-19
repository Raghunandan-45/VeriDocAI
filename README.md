# VeriDoc AI: document tampering localization (prototype)

> Status: Phase 3 done (U-Net trained on Kaggle, validated locally, thresholds frozen on
> validation). No test-split results exist yet (Phase 4).

VeriDoc AI takes a document image and outputs a pixel-level heatmap of likely edited regions,
a binary predicted mask, an overlay, and an image-level **risk score** (not a calibrated
probability). It is a research / portfolio prototype trained and evaluated on **synthetic**
documents only. It is not a fraud detector, does not detect all forgeries, and cannot prove
that a document is authentic.

## Planned components

- Synthetic dataset: Pillow-rendered fictional invoices and marksheets with two manipulation
  types (text overwrite, region splice), pixel masks and source-level train/val/test splits.
- Baseline: Error Level Analysis (ELA) anomaly score map with a validation-selected threshold.
- Model: U-Net with an ImageNet-pretrained ResNet-18 encoder (segmentation_models_pytorch),
  trained on 256x256 crops on a Kaggle T4 GPU; tiled full-page inference on CPU.
- Evaluation: thresholds and score rules selected on validation only, frozen, then applied
  once to the held-out test split, with bootstrap confidence intervals.
- Serving: FastAPI app sharing the `src/docforge` inference package; Dockerfile.

## Setup (local, CPU only)

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

## Data (Phase 1)

All data is **synthetic**: fictional invoices and marksheets rendered with Pillow. Names are
random first/last-name combinations, places and organisations are invented, and all amounts,
dates and IDs are random. Results on this data are synthetic evaluation, not evidence of
real-world performance.

### Commands

```bash
# generate the dataset (images, masks, metadata, manifests, dataset hash, counts table, contact sheet)
.venv/bin/python -m research.gen.build_dataset --config configs/data.yaml
# run all dataset tests (tiny on-the-fly dataset + the generated data/ dataset)
.venv/bin/python -m pytest -v
# verify a regenerated/uploaded copy against the canonical hash (used on Kaggle)
.venv/bin/python -m research.gen.dataset_hash --data-root data --check results/dataset_manifest_hash.txt
```

### Pipeline (identical for authentic and tampered pages)

1. Render a clean page, 1240x1754 RGB. Layout, colours, font family and sizes, and row counts vary by seed.
2. Stage-1 JPEG of the clean page, quality 70-95 (`compression.double: true`).
3. Tamper, or leave the page untouched for the authentic variant.
4. Resize by 0.8-1.2 (image bilinear or Lanczos; mask NEAREST, then re-binarised).
5. Gaussian blur with probability 0.3, sigma 0.3-1.0 (image only).
6. Final JPEG, quality 60-95. This is the saved image file.

**Why double compression:** with a single JPEG pass after editing, an edited region and the
rest of the page share one compression history, so Error Level Analysis (the baseline) has
almost no signal. A stage-1 save of the clean page before editing makes the setting
realistic (most real documents were compressed before anyone edited them) and gives ELA a
fair chance. You can switch it off with `compression.double: false`; the choice is recorded in
every sample's metadata.

### Manipulations

- **Text overwrite** (one per source): pick an editable value field (amount, quantity, date,
  name, ID, marks, grade, result, percentage). Cover it with the local background (robust
  median colour of a 4 px ring plus matching noise), then draw a new value at the same anchor.
  - **Easy:** different font family and/or a size change of at least 12%.
  - **Hard:** same family, size within ±1 px, plus a random non-empty subset of: sub-pixel or
    1-2 px offset, ink colour shift (4-10 grey levels), different anti-aliasing (supersampled or
    aliased rendering), and letter spacing.
  - Easy/hard is assigned by seed, exactly 50/50 within each split × document type.
- **Splice** (one per source): paste a rectangular patch from a donor document of the **same
  split** and same document type at a random location. Two size classes, 50/50 within each
  split × document type (recorded as `splice_size_class`):
  - **Block:** 100-420 × 40-220 px, containing ink.
  - **Field:** 1-3 text lines starting at a donor text field.

  The patch colour is shifted by the paper-colour difference (target − donor), plus a uniform
  residual of ±2 grey levels per channel, so the donor's paper tint doesn't give the edit away.

### Masks: definition and limitation

Single-channel PNG (0/255), same size as the image.
- Authentic: all zeros.
- Text overwrite: union of the cover rectangle and the new text's ink box.
- Splice: the pasted rectangle.

The mask is the **intended edit region**, not the full forensic trace. JPEG ringing, blur
halos and resampling effects that spill outside these rectangles are not labelled. Areas
inside the rectangles that look unchanged (for example blank paper inside a cover box) are
still labelled positive.

### Splits and leakage control

- Split by **source document** before generating anything: 70/15/15, stratified by document type.
  That gives 140/30/30 sources per type, so 280/60/60 sources and 840/180/180 samples overall.
- All variants of a source stay in one split.
- Splice donors come from the same split.
- Training crops come only from the train split (the sampler refuses other splits).
  - "Text-dense" crops are judged on text strokes: dark pixels minus solid fills found by a 9×9
    morphological opening. Dark header bands therefore cannot make a crop text-dense on their own.
- Val/test are evaluated on full pages.

### Fonts

Six families from **only two font designs** (DejaVu and Liberation) are used for train, val and
test. Lato and Ubuntu are held out for the stretch `test_unseen` split, so that experiment
measures transfer to an **unseen font design**, not just an unseen font file. All font files
are copied into `assets/fonts/` (sha256 values and licenses are in `docs/FONTS.md`).
Amounts are written `Rs. 12,345.00` because Liberation has no ₹ glyph.

### Files and metadata

```
data/generated/{train,val,test}/<source_id>_<variant>_<k>.jpg | _mask.png | .json
data/manifests/{train,val,test}.jsonl     # one metadata record per line
data/manifests/dataset_info.json          # config, counts, library versions
```

Metadata fields:
- Identity: `sample_id`, `source_id`, `document_type`, `split`, `condition`, `image_path` / `mask_path` (relative to `data/`).
- Labels: `image_label`, `manipulation_type`, `difficulty`, `bbox_xyxy` (final image coordinates, exclusive x1/y1).
- Reproducibility: `seed` and `seed_key`, `generator_version`, `software` (Pillow, FreeType, libjpeg-turbo, numpy and OpenCV versions).
- Fonts: `font_family` (body) and `header_font_family`.
- Processing: both JPEG qualities, resize scale and interpolation, blur sigma.
- Splices: `donor_source_id`, `donor_box_xyxy`.
- `edit`: old/new value, fonts, sizes, imperfections, cover and text boxes in render coordinates.

### Reproducibility and Kaggle

Every random draw uses `np.random.default_rng([seed, *purpose_key])`, so output does not
depend on worker count or order. `results/dataset_manifest_hash.txt` holds one sha256 over
all manifest records and **decoded** image and mask pixels, plus counts and library versions.
The locally generated dataset is **canonical**. On Kaggle you can either:
- (a) regenerate with the pinned Pillow and check the hash, or
- (b) upload the local `data/` folder as a Kaggle dataset.

If the hash differs, the check exits with an error that says to use (b).

### Known generator artifacts (synthetic-data limitations)

- Splice patches are tint-matched only in mean colour. The donor's noise level and JPEG
  history still differ, which is intended forensic signal.
- Every splice patch meets an ink floor (blocks ≥ 0.05, fields ≥ 0.03 dark-pixel fraction), so no
  patch is near-blank paper. Sparse patches near the floor can still be subtle, and the whole
  rectangle is labelled positive.
- Edited values are not kept arithmetically consistent: totals no longer add up after an edit.
  The model only sees pixels, so this is not a usable cue, but it is unrealistic.
- Only two document types and a few layouts per type. Paper is uniform colour plus Gaussian
  noise, with no scanner or photo effects beyond resize, blur and JPEG.

## Baseline and evaluation harness (Phase 2)

```bash
# ELA on validation: selects + freezes thresholds -> results/val_ela.json, results/frozen_ela.json
.venv/bin/python -m research.evaluate --method ela --split val
# test splits are refused unless a validation-frozen config is given (Phase 4)
.venv/bin/python -m research.evaluate --method ela --split test --frozen-config results/frozen_ela.json
```

**ELA anomaly score map (not a probability).**
1. Re-save the image as JPEG at quality 90 and take the absolute per-channel difference.
2. Take the max over the three channels.
3. Apply a Gaussian blur with sigma 2.
4. Score = clip(diff / 10, 0, 1).

The normalisation constant 10 is **fixed** (set from the maximum smoothed difference on 24
train pages, which was 9.5). It is not per-image min-max, which would stretch an authentic
page's noise to full range. ELA runs at native resolution because resampling destroys the
JPEG block grid.

**Shared inference code (`src/docforge`).**
- Tiled inference: tile 256, stride 128, reflect padding, 2-D Hann blending with a 0.01 floor,
  long side capped at 2000 px.
- Binned thresholds (1000 bins).
- Metrics and bootstrap CIs.

The serving API in Phase 5 will reuse the same code.

**Protocol (validation only).**
1. One global pixel threshold maximising pooled pixel F1.
2. An image-score rule (max / mean of top 500 / fraction above the pixel threshold), chosen by ROC-AUC.
3. An image threshold reaching 90% recall on tampered validation pages.

All three are frozen in a JSON file. `research/evaluate.py` refuses any split named `test*`
without such a file, and refuses files not selected on `val`.

**ELA on validation (SYNTHETIC data; thresholds selected on the same data, so optimistic).**
Source: `results/val_ela.json`. 180 pages (120 tampered, 60 authentic); 95% bootstrap CIs over 1000 resamples.

| Metric | Value |
|---|---|
| Image ROC-AUC (rule `max`) | 0.507 [0.426, 0.597] |
| Pooled pixel F1 at pixel threshold 0.144 | 0.020 [0.015, 0.026] |
| Per-image pixel F1 (tampered pages) | 0.031 [0.023, 0.039] |
| Validation recall / FPR at the frozen image threshold | 0.90 / 0.87 |

**ELA is at chance level on this dataset** (validation image AUC 0.507, CI includes 0.5).

What explains it, according to `results/ela_diagnostic.json` (40 train sources, the same edits
regenerated in memory, oracle threshold, no CIs):

| Post-processing variant | Image AUC (max) | Oracle pooled pixel F1 |
|---|---|---|
| standard (resize + blur + final JPEG) | 0.513 | 0.035 |
| no resize, no blur | 0.557 | 0.043 |
| no resize, no blur, single JPEG | 0.529 | 0.032 |

Removing the resize and blur that follow the edit helps only slightly. **ELA is weak even
without them**, so resampling is not the main reason. Hypothesis, not established: ELA
responds mainly to high-contrast text edges, which cover the whole page, so edited text looks
like any other text. Synthetic data only.

## Model and training (Phase 3)

- **Model:** `smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)` with ImageNet normalisation.
- **Loss:** BCE + soft Dice (weights 1 + 1).
- **Optimiser:** AdamW (lr 3e-4, weight decay 1e-4) with a cosine schedule. AMP is on when running on GPU.
- **Batching:** 16 crops of 256×256 per step, built from **page visits**. Each data item decodes one train page and
  cuts 4 crops from it, so JPEG decoding is shared across crops. A tampered page is visited with
  probability 0.75 and each crop on it is positive with probability 2/3. This keeps the crop rules:
  50% positive crops, and half of the negatives from authentic pages.
- **Epoch:** 1600 page visits, i.e. 6400 crops.
- **Validation and checkpoints:**
  - After each epoch, validate on a fixed subset of 36 full validation pages at stride 256.
  - Keep the best checkpoint by pooled pixel F1 on that subset.
  - At the end, validate the best checkpoint on all validation pages at stride 128.
- These are starting guesses, not tuned values.
- Training runs on a Kaggle T4 (see `docs/KAGGLE.md` and `research/kaggle_run.py`); inference works on CPU.

```bash
.venv/bin/python -m research.train --smoke --allow-cpu --encoder-weights none   # CPU overfit check
python -m research.train --data-root <data> --output-dir <out> [--resume] [--batch-size N] [--no-amp]
.venv/bin/python -m research.verify_artifacts --dir artifacts                  # after copying Kaggle outputs
.venv/bin/python -m research.evaluate --method model --checkpoint artifacts/best.pt --split val
```

**CPU smoke test** (`results/smoke_cpu/smoke_losses.json`): random-init encoder, 8 fixed crops,
20 steps at lr 1e-3. Loss went from 1.418 to 0.359; mean of the first 5 steps 1.178, mean of
the last 5 steps 0.403. This only shows the training loop can fit data; it is not a result.

### Training run (Kaggle T4)

The run was done remotely by the author. Its logs are in `results/kaggle_run/`, and the
sha256 of every downloaded file was re-checked locally (`results/artifacts_verification.json`).
- Data: the Kaggle copy passed the dataset hash check (`ed41f4f3…`, 840/180/180).
- Setup: 10 epochs, batch 16, AMP, ImageNet encoder.
- Speed: about 81–85 s per epoch.
- Losses: training loss fell from 1.097 to 0.423. Val-subset pooled F1 (36 pages, stride 256)
  went from 0.040 to a best of 0.502 at **epoch 8**, then 0.466 at epoch 10.
- `best.pt` is the epoch-8 checkpoint. It loads and runs on CPU (about 23.5 s for one 1906×1348
  page at stride 128).

### U-Net vs ELA on validation

**Synthetic data. Thresholds and score rules were selected on these same validation pages, so
every number here is optimistic.** The frozen held-out test results come in Phase 4.
Sources: `results/val_model.json`, `results/val_ela.json`, `results/val_comparison.csv`,
`results/frozen_config.json`. 180 pages (120 tampered, 60 authentic), 95% bootstrap CIs.

| Metric | ELA | U-Net (epoch 8) |
|---|---|---|
| Pixel threshold (val-selected) | 0.144 | 0.867 |
| Pooled pixel F1 | 0.020 [0.015, 0.026] | 0.602 [0.521, 0.668] |
| Pooled pixel IoU | 0.010 | 0.431 |
| Per-image pixel F1 (tampered pages) | 0.031 [0.024, 0.039] | 0.304 [0.235, 0.369] |
| Image score rule (val-selected) | max | mean of top 500 |
| Image ROC-AUC | 0.507 [0.426, 0.597] | 0.704 [0.628, 0.775] |
| Val recall / FPR at the frozen image threshold (90% recall target) | 0.90 / 0.87 | 0.90 / 0.77 [0.67, 0.87] |

The pixel scores and the image risk score are model scores, **not calibrated probabilities**.

**Breakdown for the U-Net.** Each tampered group is scored against all 60 authentic pages for AUC.

| Tampered group | n | Pooled pixel F1 | Per-image F1 | Image AUC |
|---|---|---|---|---|
| Splice (all) | 60 | 0.667 [0.587, 0.739] | 0.583 | 0.922 |
| Splice, block | 30 | 0.666 | 0.598 | 0.955 |
| Splice, field | 30 | 0.670 | 0.568 | 0.888 |
| Text overwrite (all) | 60 | 0.068 [0.000, 0.142] | 0.024 | 0.486 [0.380, 0.593] |
| Text overwrite, easy | 30 | 0.112 | 0.049 | 0.556 |
| Text overwrite, hard | 30 | 0.000 | 0.000 | 0.417 |

What this shows, on synthetic data only:
- **Splices are localised well.** Pooled F1 is about 0.67 and image AUC 0.92.
- **Text overwrites are essentially not detected.** Hard overwrites (same font, ±1 px size) get
  F1 0.000, and the overwrite AUC is at chance. The overall image AUC of 0.70 mixes
  these two very different behaviours.
- **The image-level operating point is not useful yet.** At 90% validation recall, 77% of
  authentic pages are flagged, because the model gives some high pixel scores on untouched
  pages: 37% of authentic pages have at least one pixel above the pixel threshold
  (`authentic_pages` in `results/val_model.json`).

See `docs/FONTS.md` for fonts and licenses. Sections on the model results,
limitations, privacy and deployment are added as later phases complete.
