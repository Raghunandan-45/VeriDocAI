"""Dataset fingerprint: one sha256 over manifests AND decoded pixels.

    # write (done automatically by build_dataset)
    .venv/bin/python -m research.gen.dataset_hash --data-root data --write results/dataset_manifest_hash.txt
    # verify a regenerated or uploaded copy (e.g. on Kaggle)
    python -m research.gen.dataset_hash --data-root /kaggle/working/data --check results/dataset_manifest_hash.txt

For each split, for each sample in sample_id order, the hash absorbs: the canonical JSON
manifest line, the decoded image array (shape + bytes) and the decoded mask array. Decoded
pixels are hashed rather than file bytes so the check is about content, not encoder metadata.
The locally generated dataset is canonical; a mismatch means "this copy is not that dataset".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MISMATCH_HELP = (
    "DATASET HASH MISMATCH. The regenerated data is NOT pixel-identical to the canonical local "
    "dataset (different Pillow/FreeType/libjpeg/numpy/OpenCV build or font files). Do not train on "
    "it. Use option (b): upload the locally generated data/ folder as a Kaggle dataset and point "
    "data_root at it (see docs/KAGGLE.md)."
)


def read_manifest(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_dataset_hash(data_root: Path, splits) -> dict:
    data_root = Path(data_root)
    h = hashlib.sha256()
    counts = {}
    for split in splits:
        recs = sorted(read_manifest(data_root / "manifests" / f"{split}.jsonl"), key=lambda r: r["sample_id"])
        counts[split] = len(recs)
        h.update(f"split={split};n={len(recs)}\n".encode())
        for r in recs:
            h.update(json.dumps(r, sort_keys=True).encode())
            for key in ("image_path", "mask_path"):
                arr = np.asarray(Image.open(data_root / r[key]))
                h.update(str(arr.shape).encode())
                h.update(np.ascontiguousarray(arr).tobytes())
    return {"sha256": h.hexdigest(), "counts": counts, "num_samples": sum(counts.values())}


def write_hash_file(path: Path, info: dict) -> None:
    from research.gen.build_dataset import software_versions
    lines = [f"sha256={info['sha256']}", f"num_samples={info['num_samples']}"]
    lines += [f"count_{s}={n}" for s, n in info["counts"].items()]
    lines += [f"{k}={v}" for k, v in software_versions().items()]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def parse_hash_file(path: Path) -> dict:
    return dict(line.split("=", 1) for line in Path(path).read_text().splitlines() if "=" in line)


def check(data_root: Path, hash_file: Path) -> dict:
    """Raise SystemExit with a loud message if counts or the hash differ from `hash_file`."""
    ref = parse_hash_file(hash_file)
    splits = [k[len("count_"):] for k in ref if k.startswith("count_")]
    info = compute_dataset_hash(data_root, splits)
    problems = [f"count_{s}: expected {ref[f'count_{s}']}, got {n}"
                for s, n in info["counts"].items() if str(n) != ref[f"count_{s}"]]
    if info["sha256"] != ref["sha256"]:
        problems.append(f"sha256: expected {ref['sha256']}, got {info['sha256']}")
    if problems:
        raise SystemExit("\n".join(["!" * 80, MISMATCH_HELP, *problems, "!" * 80]))
    print(f"DATASET HASH OK: {info['sha256']} ({info['counts']})")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write")
    g.add_argument("--check")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    a = ap.parse_args()
    if a.check:
        check(Path(a.data_root), Path(a.check))
    else:
        info = compute_dataset_hash(Path(a.data_root), a.splits)
        write_hash_file(Path(a.write), info)
        print(info)


if __name__ == "__main__":
    sys.exit(main())
