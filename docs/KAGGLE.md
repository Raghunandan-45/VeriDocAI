# Training on a Kaggle T4 (via VS Code)

`research/train.py` is the source of truth. `research/kaggle_run.py` is a set of VS Code
notebook cells (`# %%`) that call it on the remote kernel. Nothing here has been run on
Kaggle by the assistant; you run it, then copy the outputs back for local verification.

**Secrets:** never put the kernel link, a token or a private repo URL in a tracked file. If you
clone with git, set `VERIDOC_REPO_URL` as an environment variable in the kernel session only.
Machine-local values belong in `.env` or `configs/local.yaml` (both gitignored).

## 0. Prepare uploads (locally)

The locally generated `data/` is the **canonical dataset** (sha256 in
`results/dataset_manifest_hash.txt`). There are two ways to get data onto Kaggle:

- **(b) Recommended: upload the data.** Zip it and upload it as a private Kaggle dataset:
  ```bash
  cd data && zip -qr ../veridoc_data.zip generated manifests && cd ..
  ```
- **(a) Alternative: regenerate on the kernel.** Uses pinned Pillow/FreeType. The hash check
  fails loudly if the pixels differ; if it fails, use (b).

You also need the code on the kernel. Either push the repo and clone it with
`VERIDOC_REPO_URL`, or zip it and upload it as a second private dataset:
```bash
zip -qr veridoc_code.zip requirements.txt configs research src assets results/dataset_manifest_hash.txt \
    -x "*__pycache__*" "configs/local.yaml" "*.pyc"
```
That is everything the kernel needs (about 3 MB, 46 files):

| Path | Why |
|---|---|
| `requirements.txt` | cell 2 installs it (without torch) |
| `configs/` (`data.yaml`, `train.yaml`, `eval.yaml`) | crop rules, training and tiling settings |
| `research/` | `train.py`, `sampler.py`, `config.py`, `gen/` (hash check, dataset generator) |
| `src/docforge/` | model, tiling, preprocessing, metrics |
| `results/dataset_manifest_hash.txt` | the canonical hash that cell 3 checks against |
| `assets/fonts/` | only needed if you regenerate the data on the kernel (option a) |

Not needed: `data/` (separate dataset), `tests/`, `docs/`, `artifacts/`, `.venv/`, other `results/`
files. A zip built this way was unpacked into an empty folder and verified locally. From that copy,
the dataset hash check passed and `research.train --smoke` ran.

## 1. Kernel settings

- Accelerator: **GPU T4**.
- Internet: **On** (needed for `pip install` and the ImageNet ResNet-18 weights).
- Add **every** dataset from step 0 (code and data) to the notebook with **Add Input**. Uploading a
  dataset to your account does not mount it. Only added inputs appear under `/kaggle/input`.
- Kaggle extracts uploaded zips. Current kernels mount datasets at
  `/kaggle/input/datasets/<owner>/<slug>/`; older kernels used `/kaggle/input/<slug>/`. You don't
  need the exact path: cells 2 and 3 search all of `/kaggle/input`.

## 2. Run the cells in `research/kaggle_run.py`

| Cell | What it does | What to set |
|---|---|---|
| 1 | Prints Python/torch versions, `torch.cuda.is_available()`, GPU name and free disk. Stops if there is no GPU. | `ALLOW_CPU` (keep False) |
| 2 | Finds the code by searching `REPO_INPUT` and then all of `/kaggle/input` recursively for `research/train.py`, so nested zip folders are fine. Otherwise it `git clone`s `VERIDOC_REPO_URL`. If neither works it prints what is mounted under `/kaggle/input` and stops. Then it installs `requirements.txt` **without torch/torchvision** (the kernel's CUDA torch is kept) and prints library versions. | `REPO_INPUT` (any folder at or above the code) or env `VERIDOC_REPO_URL` |
| 3 | Finds the uploaded data (the first folder containing both `manifests/train.jsonl` and `generated/`) or regenerates it. Prints manifest counts and runs `research.gen.dataset_hash --check`, which **fails loudly** on any mismatch. | `DATA_MODE`, `DATA_INPUT` |
| 4 | Smoke run (20 steps on 8 fixed crops; must show a decreasing loss), then full training to `/kaggle/working/run`. | `BATCH_SIZE`, `EPOCHS`, `AMP`, `ENCODER_WEIGHTS`, `WEIGHTS_PATH`, `RESUME` |
| 5 | Prints sha256 and size of every output file and writes `/kaggle/working/veridoc_artifacts.zip`. | none |

The cells use the same commands you could type directly:
```bash
python -m research.gen.dataset_hash --data-root /kaggle/input/datasets/<owner>/<slug>/data --check results/dataset_manifest_hash.txt
python -m research.train --smoke --data-root /kaggle/input/datasets/<owner>/<slug>/data --output-dir /kaggle/working/smoke
python -m research.train --data-root /kaggle/input/datasets/<owner>/<slug>/data --output-dir /kaggle/working/run \
       --batch-size 16 --epochs 10 --amp
```

**Timing:** no timing estimate is given here. After epoch 1, read `epoch_seconds` in
`/kaggle/working/run/metrics.csv` and decide whether 10 epochs fit your session.

**Disconnects:** every epoch writes `last.pt` (model, optimizer, scheduler, AMP scaler) and
appends a row to `metrics.csv`. To continue, set `RESUME = True` in cell 4 and run it again
with the same `--output-dir` (or pass `--resume`). This only helps while the session is
still alive.

**Interactive sessions do not keep files.** In an interactive (draft) session, `/kaggle/working`
is wiped when the session stops, including when Kaggle stops it for inactivity. Download
anything you need from the Output panel before you stop or leave the session. For the full
training run, use **Save Version -> Save & Run All (Commit)**. That runs every cell in the
background without the browser open and keeps everything in `/kaggle/working` as the
version's Output, which you can download from the notebook's Output tab afterwards. The
cells run top to bottom unattended, so set `EPOCHS`, `BATCH_SIZE` and the other options
before you commit.

**Out of memory:** lower `BATCH_SIZE`. It must stay divisible by `train.crops_per_visit`
(4), e.g. 12 or 8.

## 2b. Run 2 (the single approved retrain)

Run 2 uses `configs/train_run2.yaml`. It is identical to `configs/train.yaml` (run 1) except
for four settings: per-crop Dice, a validation subset stratified by manipulation type,
checkpoint selection by macro F1, and 20 epochs with the cosine schedule spanning all 20.
`configs/train.yaml` still reproduces run 1.

Cell 4 is already set for run 2: `TRAIN_CONFIG = "configs/train_run2.yaml"`, `RUN_NAME = "run2"`, and
`EPOCHS = None` so the 20 epochs come from the config. Outputs go to `/kaggle/working/run2`, and
cell 5 writes `/kaggle/working/veridoc_artifacts_run2.zip`. Unzip it locally into
`artifacts/run2/` so that run 1 in `artifacts/` stays untouched. Then run:
```bash
.venv/bin/python -m research.verify_artifacts --dir artifacts/run2 --out results/artifacts_verification_run2.json
.venv/bin/python -m research.evaluate --method model --checkpoint artifacts/run2/best.pt --split val \
       --out results/val_model_run2.json --frozen-out results/frozen_config_run2.json
```

## 3. Fallback: no Internet for pretrained weights

1. On a machine with Internet, download torchvision's ImageNet ResNet-18 weights file
   (`resnet18-f37072fd.pth`, from `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`).
2. Upload it as a private Kaggle dataset and add it to the kernel.
3. In cell 4 set `ENCODER_WEIGHTS = "none"` and
   `WEIGHTS_PATH = "/kaggle/input/datasets/<owner>/<slug>/resnet18-f37072fd.pth"`. You can also set
   `model.weights_path` in `configs/train.yaml`.

`docforge.model.load_encoder_weights` loads it into the U-Net encoder (the `fc.*` keys are
ignored). It raises an error if any encoder weight is missing.

## 4. Files to download

Download `/kaggle/working/veridoc_artifacts.zip`, or these files from `/kaggle/working/run/`:

| File | Content |
|---|---|
| `best.pt` | best checkpoint by validation-subset pooled pixel F1 (model weights + metadata) |
| `last.pt` | last epoch, including optimizer/scheduler/scaler state (for resuming; larger) |
| `config_used.yaml` | resolved training config, environment, val subset ids |
| `data_config.yaml` | copy of `configs/data.yaml` used |
| `metrics.csv` | one row per epoch: losses, val-subset F1/threshold/AUC, lr, epoch seconds |
| `final_val.json` | best checkpoint on all validation pages (stride 128); optimistic |
| `SHA256SUMS.txt` | sha256 of every file above |

Also copy the cell outputs of cells 1, 3 and 4 into the chat. They are the only record of
the kernel environment and the remote hash check.

## 5. Local verification (done by the assistant after you copy the files)

```bash
unzip veridoc_artifacts.zip -d artifacts/
.venv/bin/python -m research.verify_artifacts --dir artifacts      # sha256 + CPU load + forward pass
.venv/bin/python -m research.evaluate --method model --checkpoint artifacts/best.pt --split val
.venv/bin/python -m research.compare_val                            # ELA vs model on validation
```

- `verify_artifacts` recomputes every sha256 in `SHA256SUMS.txt`, loads `best.pt` on CPU with
  `encoder_weights=None` (no network) and runs one full validation page.
- `evaluate --split val` selects and freezes the pixel threshold, image-score rule and image
  operating point into `results/frozen_config.json`. That file also records the checkpoint
  hash, train config hash and stride.
- No test split is touched in Phase 3.
