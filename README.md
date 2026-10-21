# VeriDoc AI: document tamper localization on synthetic data

> **Status:** all phases complete (research prototype). On the synthetic held-out test split the
> U-Net localises pasted regions (splices) reasonably well but mostly misses edited text, and
> its image-level score cannot be used as a filter. The local API therefore returns heatmaps
> and region boxes, never a tampered/authentic verdict.

## What this is

VeriDoc AI is a **research project on localising edits in document images, trained and
evaluated on synthetic data only**. Given a page image it produces a pixel-level score
heatmap, an overlay, boxes around high-score regions and an experimental image-level
**risk score** (not a calibrated probability).

It is **not a production verifier**. It was never tested on real documents or real forgeries,
it does not detect all kinds of edits, and it cannot show that a document is authentic.

Every number in the tables below is copied from a file in `results/` (or, for settings,
`configs/`), and `research/check_readme_numbers.py` checks that they still match. All
results are **synthetic evaluation, not evidence of real-world performance**.

**In one paragraph:** the dataset is fictional invoices and marksheets with two edit types,
text overwrites and splices. Thresholds were selected on validation and frozen, then the
held-out test split was scored once. The U-Net reaches test pooled pixel F1 0.537 against
0.023 for an ELA baseline. Splices score 0.618; text overwrites only 0.187. At the frozen
image threshold, 92% of authentic test pages are flagged.

## Architecture

```
configs/data.yaml ──► research/gen (Pillow renderer, tamperer, post-processing)
                         │  data/generated/{train,val,test}/*.jpg|_mask.png|.json
                         │  data/manifests/*.jsonl, results/dataset_manifest_hash.txt
                         ▼
research/sampler.py ──► research/train.py (256×256 crops; GPU: Kaggle T4) ──► artifacts/run2/best.pt
                                                                                 │
src/docforge  (shared: preprocess, tiled inference, ELA, thresholds, metrics, analyze)
   ▲                    ▲                                                        │
   │                    │                                                        ▼
research/evaluate.py: select on val ─► results/frozen_*.json ─► one frozen test run ─► results/test_*.json
   │
serving/app.py (FastAPI) ─► GET /health, POST /v1/analyze, upload page ─► Dockerfile (CPU)
```

- **`src/docforge`** is used by both the evaluation and the API, so the tiling, thresholding
  and scoring are the same code in both places. A test checks that the API's risk score for a
  real validation page equals the score the evaluation recorded.
- **`research/`** holds data generation, training, evaluation, failure analysis, latency and
  checks.
- **`serving/`** holds the API, the upload checks, the fixed limitations text and the static
  upload page.

## Dataset and how it is generated

Fictional documents rendered with Pillow: invoices and marksheets, 1240×1754 before resizing.
Names, organisations, places, amounts, dates and IDs are random and invented. The generator
version is `1.2.0` (`configs/data.yaml`), and the global seed is `20261016`. One sha256
(`results/dataset_manifest_hash.txt`) covers all manifest records and the **decoded** image
and mask pixels:
`ed41f4f37c3f2cb0ce53b9f8da60670952c213acf2f48520f7adad40adf5ad1c`.

Each source document gives three samples: authentic, text overwrite and splice. Counts
(`results/dataset_counts.csv`, `results/dataset_manifest_hash.txt`):

<!-- check:dataset -->
| Split | Samples | Authentic | Splice (block / field) | Text overwrite (easy / hard) |
|---|---|---|---|---|
| train | 840 | 280 | 140 / 140 | 140 / 140 |
| val | 180 | 60 | 30 / 30 | 30 / 30 |
| test | 180 | 60 | 30 / 30 | 30 / 30 |
| **total** | 1200 | 400 | 200 / 200 | 200 / 200 |

**Processing**, identical for authentic and tampered pages:
1. Render.
2. Stage-1 JPEG of the clean page (quality 70–95).
3. Edit, or leave untouched.
4. Resize ×0.8–1.2 (the mask uses nearest-neighbour).
5. Gaussian blur with probability 0.3.
6. Final JPEG (quality 60–95).

The stage-1 JPEG before editing gives ELA a fair chance: with one JPEG pass after the edit,
edited and untouched pixels share one compression history.

**Edits:**
- **Text overwrite:** cover an existing value with the local background and draw a new value.
  - *Easy:* a different font family and/or a size change of at least 12%.
  - *Hard:* same family and similar size, plus small imperfections: offset, ink-colour shift,
    changed anti-aliasing, letter spacing.
  - Easy and hard are split 50/50 by seed.
- **Splice:** paste a rectangle from a donor document of the **same split** and type.
  - *Block* or *field* sized, 50/50.
  - The paper tint is matched in mean colour.

**Masks** mark the **intended edit region**: for a text overwrite, the cover rectangle plus
the new text box; for a splice, the pasted rectangle. JPEG ringing or blur halos outside the
rectangles are not labelled. Blank paper inside a cover rectangle is labelled positive.

Known generator artifacts:
- Totals no longer add up after an edit.
- There are only two document types and two font designs: six families from DejaVu and
  Liberation. Lato and Ubuntu are reserved for an unseen-font test that was not run.
- Paper is flat colour plus noise, with no scanner or camera effects.

Fonts and licences: `docs/FONTS.md`.

## Splits and leakage controls

- **Split by source document before anything is generated:** 70/15/15, stratified by document
  type, fixed seed. All three variants of a source stay in one split.
- **Splice donors come from the same split** as the target, so no test pixels appear in
  training pages.
- **Training crops come only from the train split.** The crop sampler refuses other splits.
  Validation and test are evaluated on full pages.
- **Thresholds and score rules are selected on validation only and frozen in a JSON file.**
  `research/evaluate.py` refuses any `test*` split without a frozen config, and refuses frozen
  files that were not selected on `val` or that do not match the checkpoint, stride and
  settings.
- **The test split was scored once per method.** That is one U-Net run and one ELA run, plus
  the post-hoc robustness perturbations under the same frozen config. Nothing was changed
  afterwards.
- Tests in `tests/test_splits.py` check source isolation, same-split donors, easy/hard and
  splice-class balance, and deterministic split sizes. They run on a freshly generated tiny
  dataset and on `data/`.

## Models

### ELA baseline

Error Level Analysis, reported as an **anomaly score map**, not a probability:
1. Re-save at JPEG quality 90.
2. Take the absolute difference, then the max over channels.
3. Gaussian blur, sigma 2.
4. Divide by a **fixed** constant of 10, set from train-page maxima, and clip to [0, 1].

The pixel threshold, the image-score rule and the image threshold were all selected on
validation.

ELA is at chance level here. A diagnostic on train pages (`results/ela_diagnostic.json`,
oracle threshold, no CIs) shows that removing resize and blur barely helps:

<!-- check:ela_diag -->
| ELA post-processing variant (train pages) | Image AUC (max rule) | Oracle pooled pixel F1 |
|---|---|---|
| standard pipeline | 0.513 | 0.035 |
| no resize, no blur | 0.557 | 0.043 |
| no resize, no blur, single JPEG | 0.529 | 0.032 |

*Hypothesis, not established:* ELA responds mainly to high-contrast text edges, which cover
the whole page, so edited text looks like any other text.

### U-Net

- **Architecture:** `smp.Unet("resnet18", encoder_weights="imagenet", in_channels=3, classes=1)`
  with ImageNet normalisation.
- **Loss and optimiser:** BCE + Dice, AdamW, cosine schedule, AMP on GPU.
- **Training data:** 256×256 native-resolution crops, half of them containing an edit;
  negatives come mostly from text-dense areas. There are no flips (they would corrupt text).
- **Inference:** tiles of 256 at stride 128 with reflect padding, Hann-window blending, long
  side capped at 2000 px. This runs on CPU.
- **Validation choices:**
  - one global pixel threshold maximising pooled pixel F1;
  - image score = the score rule with the best validation AUC (mean of the top-500 pixel
    scores);
  - image threshold = the one reaching 90% recall on validation tampered pages.

### Run 1 → run 2: one pre-registered retrain

Run 1 found splices but almost no text overwrites. A diagnosis using only training and
validation data explained why (`results/diag_overwrite_*.json`):

<!-- check:diag -->
| Finding behind the retrain (no test data used) | Value | Source |
|---|---|---|
| Text overwrites: share of positive training crops | 49% | `diag_overwrite_crops.json` |
| Text overwrites: share of positive training pixels | 11% | `diag_overwrite_crops.json` |
| Text overwrites: share of tampered pixels in the checkpoint-selection subset | 8.5% | `diag_overwrite_context.json` |
| Sampled TRAINING overwrite pages with any detected pixel | 5 of 30 | `diag_overwrite_preds.json` |
| Median max score inside missed val overwrite masks | 0.010 | `diag_overwrite_preds.json` |

Run 2 was the **single approved retrain: per-crop Dice + stratified selection + 20 epochs**
(`configs/train_run2.yaml`). Three settings changed together, so any difference is the
effect of the combination, not of one change.

The decision rule was fixed **before** run 2 was trained: keep the run with the higher
full-validation macro pooled F1, the mean of splice F1 and overwrite F1. Results were to be
reported either way. Run logs: `results/kaggle_run/`, `results/kaggle_run2/`; decision:
`results/final_model.json`.

<!-- check:runs -->
| Run | Training change | Epochs | Best epoch | Median s / epoch (`T4` GPU) | Val macro pooled F1 (decision metric) |
|---|---|---|---|---|---|
| `run1` (baseline) | batch Dice, random val subset, pooled-F1 selection | 10 | 8 | 82 | 0.368 |
| **`run2` (final)** | **per-crop Dice + stratified selection + more epochs** | 20 | 18 | 97 | 0.461 |

Validation numbers (`results/val_ela.json`, `results/val_model.json`,
`results/val_model_run2.json`, `results/final_model.json`, `results/val_comparison.csv`).
**Thresholds were selected on these same pages, so these numbers are optimistic.**

<!-- check:val_models -->
| Metric (validation; thresholds selected on these pages) | ELA | U-Net `run1` | U-Net `run2` |
|---|---|---|---|
| Pixel threshold (selected on val) | 0.144 | 0.867 | 0.880 |
| **Macro pooled F1 (pre-registered decision metric)** | **0.028** | **0.368** | **0.461** |
| Pooled pixel F1, splice | 0.043 | 0.667 [0.587, 0.739] | 0.669 [0.574, 0.743] |
| Pooled pixel F1, text overwrite | 0.013 | 0.068 [0.000, 0.142] | 0.254 [0.155, 0.353] |
| └ easy | 0.012 | 0.112 | 0.236 |
| └ hard | 0.013 | 0.000 | 0.276 |
| Pooled pixel F1, all pages | 0.020 | 0.602 [0.521, 0.668] | 0.591 [0.505, 0.659] |
| Image AUC (selected score rule) | 0.507 | 0.704 [0.628, 0.775] | 0.725 [0.652, 0.792] |
| FPR at the frozen 90%-recall image threshold | 0.87 | 0.77 | 0.75 |
| Authentic pages with any pixel above the pixel threshold | 0.97 | 0.37 | 0.88 |

Run 2 won the pre-registered rule and is the final model:
- **What improved:** text-overwrite F1 rose and splice F1 held.
- **What it cost:** more authentic pages carry at least one high-score pixel.

## Test results (held-out synthetic test split, frozen config, one run)

- **Model:** run 2, checkpoint epoch 18, sha256 `4fdae75a…`.
- **Frozen config:** `results/frozen_config_run2.json`: pixel threshold 0.88, image score =
  mean of the top-500 pixel scores, image threshold selected for 90% validation recall.
- **Provenance:** `results/test_model.json` and `results/test_ela.json` record the dataset,
  config and checkpoint hashes; all match the frozen files.
- **Uncertainty:** 95% bootstrap CIs over 1000 image resamples.

**ELA vs U-Net, all test pages** (`results/test_ela.json`, `results/test_model.json`; also in
`results/test_comparison.csv`):

<!-- check:test_overall -->
| Metric (test split, one frozen run each) | ELA | U-Net `run2` |
|---|---|---|
| Pages (tampered / authentic) | 120 / 60 | 120 / 60 |
| Frozen pixel threshold | 0.144 | 0.880 |
| **Pooled pixel F1** | 0.023 [0.017, 0.030] | 0.537 [0.452, 0.608] |
| Pooled pixel IoU | 0.012 | 0.367 |
| Per-image pixel F1 (tampered pages) | 0.035 [0.027, 0.045] | 0.377 [0.311, 0.440] |
| Image AUC | 0.484 [0.394, 0.578] | 0.696 [0.621, 0.773] |
| Recall at the frozen image threshold | 0.950 [0.908, 0.983] | 0.942 [0.900, 0.983] |
| **FPR at the frozen image threshold** | 0.850 [0.767, 0.933] | 0.917 [0.833, 0.983] |
| Authentic pages with any predicted pixel | 0.98 | 1.00 |

**Per manipulation type and document type** (each tampered group is scored against all 60
authentic test pages for AUC):

<!-- check:test_groups -->
| Tampered group (test) | n | U-Net pooled F1 | U-Net per-image F1 | U-Net image AUC | U-Net recall at frozen threshold | ELA pooled F1 |
|---|---|---|---|---|---|---|
| Splice | 60 | 0.618 [0.526, 0.697] | 0.592 | 0.867 [0.790, 0.934] | 0.967 | 0.053 |
| └ block | 30 | 0.603 [0.501, 0.702] | 0.608 | 0.903 [0.807, 0.977] | 0.967 | 0.072 |
| └ field | 30 | 0.674 [0.581, 0.753] | 0.576 | 0.831 [0.709, 0.931] | 0.967 | 0.028 |
| Text overwrite | 60 | 0.187 [0.097, 0.288] | 0.161 | 0.525 [0.417, 0.629] | 0.917 | 0.011 |
| └ easy | 30 | 0.274 [0.119, 0.416] | 0.195 | 0.512 [0.374, 0.634] | 0.900 | 0.009 |
| └ hard | 30 | 0.099 [0.046, 0.179] | 0.127 | 0.537 [0.416, 0.660] | 0.933 | 0.013 |
| Invoices (both types) | 60 | 0.600 [0.487, 0.686] | 0.391 | 0.667 [0.558, 0.768] | 0.900 | 0.043 |
| Marksheets (both types) | 60 | 0.548 [0.417, 0.662] | 0.362 | 0.725 [0.628, 0.817] | 0.983 | 0.025 |

**Robustness check.** Each perturbation is applied alone to the saved test images, with
parameters drawn per page from a fixed seed (`configs/robustness.yaml`, chosen outside the
training ranges). The frozen config is used unchanged. Sources:
`results/test_model_perturb_*.json` and `results/test_robustness.csv`.

<!-- check:robustness -->
| Metric (test, frozen config) | Clean | JPEG re-encode, quality 40–60 | Downscale ×0.60–0.75 | Blur σ 1.2–1.6 |
|---|---|---|---|---|
| Pooled pixel F1, all | 0.537 [0.452, 0.608] | 0.507 [0.426, 0.585] | 0.394 [0.296, 0.478] | 0.330 [0.248, 0.406] |
| └ splice | 0.618 | 0.581 | 0.567 | 0.499 |
| └ text overwrite | 0.187 | 0.162 | 0.077 | 0.022 |
| Image AUC | 0.696 [0.621, 0.773] | 0.658 [0.579, 0.739] | 0.653 [0.577, 0.732] | 0.629 [0.543, 0.711] |
| Recall / FPR at the frozen image threshold | 0.942 / 0.917 | 0.908 / 0.883 | 0.933 / 0.933 | 0.983 / 0.950 |
| Flagged tampered pages with no correct pixel | 36 of 113 | 43 of 109 | 42 of 112 | 49 of 118 |

How to read the test results:
- **Splices are localised; text overwrites mostly are not.** The overwrite image AUC is close
  to 0.5.
- **The image-level score does not work as a filter.** At the threshold frozen for 90%
  validation recall, 92% of authentic test pages are flagged.
- **High recall overstates localisation.** Many flagged tampered pages have no correctly
  predicted pixel: their score came from false positives elsewhere on the page.
- **Blur removes most of the overwrite signal.** Splices degrade gradually.

## Failure analysis (summary)

Full write-up with 12 annotated test examples: `docs/FAILURE_ANALYSIS.md`. Figure:
`docs/figures/failure_examples.jpg`. All explanations there are labelled hypotheses.

**Where false positives fall** on authentic **validation** pages
(`results/failure_regions_val.json`; the region classes come from a simple pixel-colour
rule). The answer is ordinary text, not headers or table lines:

<!-- check:fp_regions -->
| Region class (authentic val pages) | Share of false-positive pixels | Share of page area | Lift (FP share / area share) |
|---|---|---|---|
| text | 86.2% | 8.8% | 9.81 |
| blank paper | 11.6% | 80.8% | 0.14 |
| table / rule lines | 1.6% | 5.7% | 0.28 |
| solid fill (header bands, shaded cells) | 0.6% | 4.8% | 0.13 |
| header zone (`top 15%` of the page) | 10.8% |  |  |

**Hard vs easy overwrites.** On validation, hard overwrites scored slightly above easy ones.
That reversed on test, and the validation CIs overlap almost completely, so the gap was
noise (`results/overwrite_imperfections.json`, `results/val_model_run2.json`,
`results/test_model.json`):

<!-- check:easy_hard -->
| Split | Overwrite difficulty | n | Pages with any correct pixel | Pooled pixel F1 [95% CI] |
|---|---|---|---|---|
| val | easy | 30 | 0.40 | 0.236 [0.117, 0.383] |
| val | hard | 30 | 0.43 | 0.276 [0.135, 0.417] |
| test | easy | 30 | 0.47 | 0.274 [0.119, 0.416] |
| test | hard | 30 | 0.30 | 0.099 [0.046, 0.179] |

**Which generator imperfections go with detection.** Fraction of hard overwrite pages with any
correctly predicted pixel (`results/overwrite_imperfections.json`). Imperfections overlap on
the same pages, the groups are small and there are no CIs:

<!-- check:imperfections -->
| Hard overwrites | val: with | val: without | test: with | test: without |
|---|---|---|---|---|
| changed anti-aliasing | 0.58 (n 19) | 0.18 (n 11) | 0.40 (n 15) | 0.20 (n 15) |
| ink colour shift | 0.52 (n 21) | 0.22 (n 9) | 0.35 (n 17) | 0.23 (n 13) |
| letter spacing | 0.29 (n 17) | 0.62 (n 13) | 0.33 (n 12) | 0.28 (n 18) |
| small offset | 0.47 (n 17) | 0.38 (n 13) | 0.24 (n 17) | 0.38 (n 13) |

Only the changed anti-aliasing goes with detection on both splits. *Hypothesis:* the model
partly detects a rendering artifact of this generator, which would not transfer to real edits.

Other patterns (hypotheses, from the annotated examples):
- Untouched numeric table cells attract false positives.
- Clean size-only or font-family-only overwrites are missed although a person can see them.
- Splices on blank paper with matching background are missed; splices that cut across table
  structure are found.

## Limitations

- **Synthetic data only.** It comes from one generator, with two document types and two font
  designs. The test split shares the generator, fonts and processing ranges with training.
  **No real documents or real forgeries were tested.**
- **Text overwrites are weak:** test pooled F1 0.187, image AUC close to chance. Hard
  overwrites score lower still, and detection may rely on generator rendering artifacts.
- **Blur sensitivity:** after Gaussian blur, overwrite F1 drops close to zero.
- **The image-level score is unusable as a filter:** at the frozen operating point the test
  FPR is 0.917 (92% of authentic pages flagged). The score is not calibrated.
- **Small evaluation:** 180 test pages (groups of 30–60), so the CIs are wide. There is one
  training seed per run, and only one retrain was done.
- **Masks label the intended edit region**, not every forensic trace.
- **Not run (stretch items, skipped):** the unseen-font split, the unseen-processing split
  (`test_robust`), a hand-edited sanity set and OCR.

## The API

`serving/app.py` (FastAPI) exposes:
- `GET /health`;
- `POST /v1/analyze` (multipart field `file`, JPEG or PNG);
- a static upload page at `/`.

**It deliberately returns no tampered/authentic verdict, no risk tier and no image
threshold**, because of the test FPR above. It returns:
- a score heatmap (PNG) and an overlay (JPEG) as base64, downscaled for display;
- boxes of the connected regions at or above the frozen pixel threshold, in original pixels,
  with area and peak/mean score;
- the risk score, labelled `"experimental"`, with a note that it is not a probability and not
  a screening filter;
- a fixed `limitations` text (`serving/limitations.py`). A test checks every number in it
  against `results/test_model.json` and `results/test_model_perturb_blur.json`.

**Upload checks** (`serving/validation.py`; limits in `configs/serve.yaml` are initial
assumptions; tested in `tests/test_serving.py`):

<!-- check:inputs -->
| Upload check (`serving/validation.py`) | Limit (`configs/serve.yaml`) | Response |
|---|---|---|
| file or request body too large | 10 MiB | 413 |
| not a JPEG/PNG (judged from the bytes) | JPEG, PNG | 415 |
| decoded size, read from the header before decoding | side ≤ 8000 px, ≤ 40 MP, side ≥ 32 px | `422` |
| truncated or corrupt image data | full decode must succeed | `400` |
| checkpoint missing or `sha256` ≠ frozen config | checked at startup | `503` |

**Privacy notes:**
- **Uploads stay in memory.** Starlette spools multipart files over 1 MB to a temp file by
  default; the app raises that limit above the upload limit, and a test fails if an upload
  touches disk.
- **Nothing is stored.** No database, no files, no caching (`Cache-Control: no-store`).
- **Logs** hold a request id, byte size, image size, region count and time, never file names
  or image content (tested).
- **No network at serve time.** The model is built with `encoder_weights=None` and the
  trained weights are loaded from a local file (tested with downloads blocked; the container
  check runs with `--network none`).
- **Not handled:** EXIF orientation is not applied (as in the evaluation). There is no
  authentication or rate limiting, so this is a **local** tool, not a public service.

## How to run

**Environment and tests** (Python 3.10, CPU):

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m research.gen.build_dataset --config configs/data.yaml   # data/ (about 1200 files x 3)
.venv/bin/python -m pytest -q
.venv/bin/python -m research.check_readme_numbers                           # README tables vs results/
```

**API locally.** The checkpoint `artifacts/run2/best.pt` is gitignored:

```bash
VERIDOC_CHECKPOINT=artifacts/run2/best.pt PYTHONPATH=src:. .venv/bin/uvicorn serving.app:app --port 8000
# http://127.0.0.1:8000/  (upload page)
curl -F "file=@page.jpg;type=image/jpeg" http://127.0.0.1:8000/v1/analyze
```

**Docker (CPU-only image, no checkpoint inside):**

```bash
docker build -t veridoc-api:0.5 .
# a) mount the checkpoint read-only
docker run --rm -p 8000:8000 -v "$PWD/artifacts/run2/best.pt:/models/best.pt:ro" veridoc-api:0.5
# b) or fetch it from private storage first and check its sha256 before mounting
#    (the URL comes from your own secret store; never commit it)
curl -fL "$VERIDOC_MODEL_URL" -o best.pt
echo "4fdae75a79754102787fe037aa22bd623954ae3eeb93eef800028a3f4fd119d4  best.pt" | sha256sum -c -
docker run --rm -p 8000:8000 -v "$PWD/best.pt:/models/best.pt:ro" veridoc-api:0.5
# verify a built image (offline run, health, analysis, rejections)
.venv/bin/python -m research.docker_check          # -> results/docker_verification.json
```

At startup the app compares the checkpoint's sha256 with `results/frozen_config_run2.json`
and refuses to serve on a mismatch (`/health` returns 503). The frozen thresholds only apply
to that exact checkpoint.

<!-- check:docker -->
| Docker check (`--network none`, checkpoint mounted read-only) | Result |
|---|---|
| image size (compressed content, MB) | 380 |
| torch in image / CUDA available | `2.14.1+cpu False` |
| network inside the container | unreachable: gaierror |
| `GET /health` | 200 |
| `POST /v1/analyze` (validation splice page) | 200, 2 regions |
| largest predicted box `[x0, y0, x1, y1]` | [41, 1206, 232, 1286] |
| non-image upload | 415 |
| oversized upload (announced body over the limit) | 413 |

## Latency (CPU)

Every request goes through the full API path in-process (upload parsing, checks, decode,
tiled inference, regions, image encoding), with no network. Source: `results/latency_cpu.json`
(`.venv/bin/python -m research.latency`).

An **earlier latency run (Phase 5) was contaminated**: it overlapped Docker Desktop starting up
and other containers on the machine. It was **replaced** by a re-measurement on an otherwise
idle machine (Docker Desktop and all containers stopped), which is what the tables show.

<!-- check:latency_setup -->
| Latency setup | Value |
|---|---|
| CPU / torch | `13th Gen Intel(R) Core(TM) i5-1335U` / `2.14.1+cpu` |
| Logical CPUs / torch threads | 12 / 6 |
| RAM (GiB) | 7.6 |
| Pages timed (validation) / warm-up requests per stride | 20 / 2 |
| Page width / height range (px) | 996–1452 / 1409–2054 |
| Median page size (megapixels) | 2.16 |
| Pages downscaled to the analysis cap | 3 (cap 2000 px long side) |

**Stride trade-off.** Latency comes from the run above. The F1 columns are measured on
**validation only**: the stride-128 frozen thresholds were applied unchanged at the larger
strides with `research/evaluate.py --stride-study`, which is refused on test splits. Sources:
`results/val_model_run2.json`, `results/val_model_run2_stride192.json`,
`results/val_model_run2_stride256.json`.

<!-- check:latency -->
| Stride | Request median / `p95` (s) | Val pooled pixel F1 | Val splice F1 | Val overwrite F1 | Val FPR at frozen threshold | Authentic val pages with any predicted pixel |
|---|---|---|---|---|---|---|
| **128** (default; used on test) | 14.6 / 20.4 | 0.591 [0.505, 0.659] | 0.669 | 0.254 | 0.75 | 0.88 |
| 192 | 7.3 / 9.6 | 0.558 [0.482, 0.621] | 0.646 | 0.211 | 0.88 | 0.97 |
| 256 | 4.2 / 5.6 | 0.463 [0.388, 0.529] | 0.587 | 0.167 | 0.97 | 1.00 |

- **Default stride 128:** the API serves at 128, the only stride evaluated on test.
  `VERIDOC_STRIDE` changes it, and `/health` reports the stride in use next to the test stride.
- **Stride 192:** roughly halves the time at a small validation F1 cost with overlapping CIs.
- **Stride 256** (no tile overlap): loses more F1 and marks pixels on every authentic
  validation page.

## Deployment (optional)

This is a local research tool. **Latency on any hosted machine was not measured**; the
numbers above are from one laptop CPU. If you do run it elsewhere:
- **Image:** push `veridoc-api:0.5` to a registry (`docker tag` / `docker push`). The image
  contains no weights, so it can live in any registry you are allowed to use.
- **Model weights:** provide them at runtime as a read-only volume or an init-step download
  from private storage, with the sha256 check above. Alternatively, build a **private**
  image that adds `best.pt` in a final layer and push it only to a private registry.
- **Host:** any x86-64 CPU host with Docker. Allow at least several seconds per page, run one
  worker per container (torch already uses several threads), and size memory for the page
  score map.
- **Network:** put authentication, TLS and rate limiting in front of it. The app has none, and
  uploads are document images.
- **Retraining:** a retrained model needs a new frozen config from validation, never a reused
  threshold.

## Reproducibility

- **Data:** `research.gen.build_dataset` is deterministic. Every random draw uses
  `np.random.default_rng([seed, *purpose_key])`, so output does not depend on worker count.
  Pillow, numpy and OpenCV are pinned (`requirements.txt`; full freeze in
  `requirements-lock.txt`) because the dataset hash depends on their rendering, JPEG and
  resize behaviour. Check a copy with
  `.venv/bin/python -m research.gen.dataset_hash --data-root data --check results/dataset_manifest_hash.txt`.
- **Training (GPU):** `research/train.py` is the source of truth. It supports
  `--config configs/train_run2.yaml --data-root <data> --output-dir <out>`, `--resume`,
  `--batch-size` and `--no-amp`, and it checkpoints every epoch. A CPU smoke test is
  `--smoke --allow-cpu --encoder-weights none`.
- **Remote GPU workflow (generic):** training ran on a hosted notebook GPU (an NVIDIA T4).
  `research/kaggle_run.py` holds the notebook cells:
  1. environment check;
  2. get the repo onto the machine (git clone or an uploaded archive);
  3. regenerate or upload the data and verify the dataset hash;
  4. smoke run, then full training;
  5. print the sha256 of every output.

  `docs/KAGGLE.md` lists the steps and the files to download. Kernel links, URLs and tokens
  are never stored in the repo; the code reads them from untracked files (`.env`,
  `configs/local.yaml`). After download, `research.verify_artifacts` re-checks the sha256
  values and loads the checkpoint on CPU.
- **Evaluation:**
  1. `research.evaluate --split val` selects and freezes;
  2. `research.select_final` applies the pre-registered run rule;
  3. `research.evaluate --split test --frozen-config …` scores the test split once;
  4. `research.compare_test` and `research.failure_analysis` build the tables and the
     failure analysis.

## Repository layout

```
configs/      data, train (run 1, run 2), eval, robustness, serve (all defaults = initial assumptions)
research/     gen/ (dataset), sampler, train, evaluate, perturb, failure_analysis, latency, checks
src/docforge/ shared inference: preprocess, tiling, model, ELA, scoring, analyze
serving/      FastAPI app, upload validation, limitations text, static page
tests/        pytest suite (data, splits, crops, tiling, scoring, model, run 2, perturbations, serving)
results/      every reported number (JSON/CSV); large images are gitignored
docs/         FONTS.md, KAGGLE.md, FAILURE_ANALYSIS.md, INTERVIEW_NOTES.md, figures/
```
