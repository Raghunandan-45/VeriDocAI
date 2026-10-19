# %% [markdown]
# # VeriDoc AI - Kaggle GPU training runbook (run cell by cell on a Kaggle T4 kernel)
#
# The notebook only orchestrates: every heavy step runs as a subprocess
# (`python -m research.<module>`), so package upgrades in cell 2 take effect without a
# kernel restart. research/train.py is the source of truth; this file only calls it.
# Never paste tokens or private URLs into tracked files. See docs/KAGGLE.md.

# %% Cell 1 - environment check
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ALLOW_CPU = False  # set True only to debug the runbook without a GPU

import torch  # noqa: E402

print("python", platform.python_version())
print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
print("torch.cuda.is_available():", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
total, used, free = shutil.disk_usage("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
print(f"free disk: {free / 1e9:.1f} GB of {total / 1e9:.1f} GB")
if not torch.cuda.is_available() and not ALLOW_CPU:
    raise SystemExit("NO GPU: enable a GPU accelerator (T4) in the kernel settings, or set ALLOW_CPU = True.")

# %% Cell 2 - bootstrap: get the repo and install requirements (torch is NOT reinstalled)
WORK = Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path.cwd()
REPO_DIR = WORK / "veridocAI"
# Option A: code uploaded as a Kaggle dataset (zip is auto-extracted by Kaggle).
# Any folder at or ABOVE the code works: it is searched recursively for research/train.py,
# then the whole of /kaggle/input is searched. "" = search /kaggle/input only.
REPO_INPUT = ""                    # e.g. "/kaggle/input/datasets/<owner>/veridoc-code" (read-only; copied to REPO_DIR)
# Option B: git clone. Set the URL in the kernel only (environment variable), never in a tracked file.
REPO_URL = os.environ.get("VERIDOC_REPO_URL", "")
INSTALL_PINNED = True              # install pinned pillow/numpy/opencv/... from requirements.txt


def run(cmd, cwd=None):
    """Run a command and stream its output into this cell; raise on failure."""
    print("$", " ".join(str(c) for c in cmd), flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="", flush=True)
    if p.wait() != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}")


def find_dirs_containing(rel: str, roots, max_depth: int = 6) -> list:
    """Directories under `roots` (searched in order, up to max_depth levels) that contain the
    relative path `rel`; within each root the shallowest match comes first."""
    hits = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        found = []
        for dirpath, dirnames, _ in os.walk(root):
            d = Path(dirpath)
            if (d / rel).exists():
                found.append(d)
            if len(d.relative_to(root).parts) >= max_depth:
                dirnames[:] = []
            dirnames[:] = [n for n in dirnames if n not in ("generated", ".git", "__pycache__")]
        hits += [h for h in sorted(found, key=lambda p: len(p.parts)) if h not in hits]
    return hits


def show_input_tree(root: str = "/kaggle/input", max_depth: int = 3, max_entries: int = 12) -> None:
    """Print what is actually mounted (folders to max_depth, a few files each) to debug paths."""
    print(f"--- contents of {root} (depth <= {max_depth}) ---")
    if not os.path.isdir(root):
        print(f"{root} does not exist")
        return
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        depth = len(d.relative_to(root).parts)
        dirnames.sort()
        print("  " * depth + f"{d.name or root}/  ({len(filenames)} files)")
        for f in sorted(filenames)[:3]:
            print("  " * (depth + 1) + f)
        if depth >= max_depth or len(dirnames) > max_entries:
            if dirnames:
                print("  " * (depth + 1) + f"... {len(dirnames)} subfolders not shown")
            dirnames[:] = []


def input_roots(preferred: str) -> list:
    return [r for r in (preferred, "/kaggle/input") if r]


if not (REPO_DIR / "research" / "train.py").exists():
    hits = find_dirs_containing("research/train.py", input_roots(REPO_INPUT))
    if hits:
        src = hits[0]
        if REPO_INPUT and not str(src).startswith(REPO_INPUT):
            print(f"note: code not found under REPO_INPUT={REPO_INPUT}; using {src} instead")
        shutil.copytree(src, REPO_DIR, ignore=shutil.ignore_patterns("data", ".git", ".venv", "artifacts",
                                                                     "__pycache__"))
        print("copied code from", src)
    elif REPO_URL:
        run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])  # URL is not printed by us
    else:
        show_input_tree()
        raise SystemExit("Repo not found: no folder containing research/train.py under "
                         f"{input_roots(REPO_INPUT)}. Upload the code zip as a Kaggle dataset and add it to "
                         "this kernel (see the listing above), or set VERIDOC_REPO_URL.")
os.chdir(REPO_DIR)
print("repo:", REPO_DIR)

if INSTALL_PINNED:
    reqs = [l for l in (REPO_DIR / "requirements.txt").read_text().splitlines()
            if l.strip() and not l.lstrip().startswith(("#", "torch==", "torchvision=="))]
    (WORK / "requirements_no_torch.txt").write_text("\n".join(reqs) + "\n")
    # several OpenCV wheels provide `cv2`; keep only the pinned headless one
    run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "opencv-python", "opencv-contrib-python",
         "opencv-python-headless"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", WORK / "requirements_no_torch.txt"])
run([sys.executable, "-c", "import torch, PIL, numpy, cv2, segmentation_models_pytorch as s; "
                           "from PIL import features; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(),"
                           " '| pillow', PIL.__version__, 'freetype', features.version('freetype2'), '| numpy', "
                           "numpy.__version__, '| cv2', cv2.__version__, '| smp', s.__version__)"])

# %% Cell 3 - data: uploaded copy (recommended) or deterministic regeneration, then hash check
DATA_MODE = "uploaded"             # "uploaded" (data/ uploaded as a Kaggle dataset) or "regenerate"
DATA_INPUT = ""                    # folder at or above manifests/ + generated/; "" = search /kaggle/input
if DATA_MODE == "uploaded":
    hits = [h for h in find_dirs_containing("manifests/train.jsonl", input_roots(DATA_INPUT))
            if (h / "generated").is_dir()]
    if not hits:
        show_input_tree()
        raise SystemExit("Uploaded data not found: no folder containing manifests/train.jsonl AND generated/ "
                         f"under {input_roots(DATA_INPUT)} (see the listing above).")
    DATA_ROOT = hits[0]
else:
    DATA_ROOT = WORK / "data"
    run([sys.executable, "-m", "research.gen.build_dataset", "--config", "configs/data.yaml",
         "--data-root", DATA_ROOT, "--no-report"])
print("DATA_ROOT:", DATA_ROOT)
for split in ("train", "val", "test"):
    n = sum(1 for _ in open(DATA_ROOT / "manifests" / f"{split}.jsonl"))
    print(f"manifest {split}: {n} samples")
# Fails loudly (exit 1) if counts or pixels differ from the canonical local dataset.
# If this fails in "regenerate" mode: switch to DATA_MODE = "uploaded" (option b).
run([sys.executable, "-m", "research.gen.dataset_hash", "--data-root", DATA_ROOT,
     "--check", "results/dataset_manifest_hash.txt"])

# %% Cell 4 - smoke run, then full training (resumable)
OUT = WORK / "run"
BATCH_SIZE = 16                    # initial assumption; lower if CUDA runs out of memory
EPOCHS = 10                        # initial assumption; measure the first epoch before relying on it
AMP = True
ENCODER_WEIGHTS = "imagenet"       # needs Internet ON; offline: "none" + WEIGHTS_PATH below
WEIGHTS_PATH = ""                  # e.g. "/kaggle/input/datasets/<owner>/resnet18-weights/resnet18-f37072fd.pth"
RESUME = False                     # True after a disconnect: continues from OUT/last.pt

common = ["--data-root", DATA_ROOT, "--encoder-weights", ENCODER_WEIGHTS, "--amp" if AMP else "--no-amp"]
if WEIGHTS_PATH:
    common += ["--weights-path", WEIGHTS_PATH]
if not RESUME:
    run([sys.executable, "-m", "research.train", "--smoke", "--output-dir", WORK / "smoke", *common])
print("Starting full training. Note the 'epoch_seconds' of epoch 1 in metrics.csv before planning more epochs.")
run([sys.executable, "-m", "research.train", "--output-dir", OUT, "--batch-size", BATCH_SIZE,
     "--epochs", EPOCHS, *common, *(["--resume"] if RESUME else [])])

# %% Cell 5 - outputs to download (sha256 + size), bundled into one zip
import hashlib  # noqa: E402
import zipfile  # noqa: E402

wanted = ["best.pt", "last.pt", "config_used.yaml", "data_config.yaml", "metrics.csv", "final_val.json",
          "SHA256SUMS.txt"]
for name in wanted:
    p = OUT / name
    if p.exists():
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name:20s} {p.stat().st_size:>12,d} bytes  sha256 {h}")
    else:
        print(f"{name:20s} MISSING")
print(open(OUT / "metrics.csv").read())
bundle = WORK / "veridoc_artifacts.zip"
with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_STORED) as z:
    for name in wanted:
        if (OUT / name).exists():
            z.write(OUT / name, arcname=name)
print("download:", bundle, f"({bundle.stat().st_size / 1e6:.0f} MB) -> unzip into artifacts/ locally")



# %%
