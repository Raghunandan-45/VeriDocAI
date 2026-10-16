# VeriDoc AI: document tampering localization (prototype)

> Status: Phase 1 (synthetic dataset). No model or baseline results exist yet.

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
    1-2 px offset, ink colour shift, different anti-aliasing (supersampled or aliased
    rendering), and letter spacing.
  - Easy/hard is assigned by seed, exactly 50/50 within each split × document type.
- **Splice** (one per source): paste a rectangular patch (100-420 × 40-220 px, containing
  ink) from a donor document of the **same split** and same document type at a random location.

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

- Splice patches keep the donor's paper tint and noise, so a patch pasted onto blank paper
  shows as a faintly different rectangle. A model may learn this as a shortcut.
- Edited values are not kept arithmetically consistent: totals no longer add up after an edit.
  The model only sees pixels, so this is not a usable cue, but it is unrealistic.
- Only two document types and a few layouts per type. Paper is uniform colour plus Gaussian
  noise, with no scanner or photo effects beyond resize, blur and JPEG.

See `docs/FONTS.md` for fonts and licenses. Sections on the baseline, model, results,
limitations, privacy and deployment are added as later phases complete.
