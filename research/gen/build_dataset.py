"""Build the synthetic tampering dataset.

    .venv/bin/python -m research.gen.build_dataset --config configs/data.yaml

Steps: plan sources -> split by SOURCE (stratified by document type) -> for each source:
render, stage-1 JPEG, make variants (authentic / text overwrite / splice), post-process,
save image (.jpg), mask (.png), metadata (.json) -> write manifests -> dataset hash, counts
table and contact sheet.

Determinism: every random draw comes from `np.random.default_rng([seed, *key])` where the
key names the source and the purpose (layout, stage-1 JPEG, variant edit, post-processing,
donor choice, split). Results therefore do not depend on worker count or processing order.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import PIL
from PIL import Image, features

from research.config import load_config, resolve
from research.gen.fonts import FontRegistry
from research.gen.postprocess import (apply_post, decode_image, encode_jpeg, jpeg_roundtrip,
                                      sample_post_params)
from research.gen.tamper_ops import splice, text_overwrite
from research.gen.templates import RENDERERS

# Stream ids for seeding (never renumber: doing so changes the dataset).
S_LAYOUT, S_STAGE1, S_VARIANT, S_POST, S_DONOR, S_SPLIT, S_DIFFICULTY, S_SPLICE_CLASS = 0, 1, 2, 3, 4, 5, 6, 7
VARIANT_CODES = {"authentic": 0, "text_overwrite": 1, "splice": 2}
DOC_TYPE_CODES = {"invoice": 0, "marksheet": 1}
CORE_SPLITS = ("train", "val", "test")


def stream(seed: int, *key: int) -> np.random.Generator:
    """Independent, reproducible random stream for a given purpose key."""
    return np.random.default_rng([int(seed), *[int(k) for k in key]])


def software_versions() -> dict:
    """Library versions that affect rendered/encoded pixels (recorded in every sample)."""
    return {"pillow": PIL.__version__, "freetype": features.version("freetype2"),
            "libjpeg_turbo": features.version("jpg"), "numpy": np.__version__,
            "opencv": cv2.__version__}


def plan_sources(cfg: dict) -> list[dict]:
    """List every source document: id, type and index (used as seed key)."""
    out = []
    for doc_type, n in cfg["sources"].items():
        for i in range(int(n)):
            out.append({"source_id": f"{doc_type}_{i:04d}", "document_type": doc_type,
                        "type_code": DOC_TYPE_CODES[doc_type], "index": i})
    return out


def assign_splits(sources: list[dict], cfg: dict) -> dict[str, str]:
    """Assign each SOURCE to train/val/test, stratified by document type.

    n_val = round(n * val_frac), n_test = round(n * test_frac), the rest is train
    (round half up, so 200 sources -> 140/30/30)."""
    fr = cfg["splits"]
    split_of = {}
    for doc_type, code in DOC_TYPE_CODES.items():
        ids = [s["source_id"] for s in sources if s["document_type"] == doc_type]
        if not ids:
            continue
        order = stream(cfg["seed"], S_SPLIT, code).permutation(len(ids))
        n_val = int(np.floor(len(ids) * fr["val"] + 0.5))
        n_test = int(np.floor(len(ids) * fr["test"] + 0.5))
        for rank, j in enumerate(order):
            split_of[ids[j]] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"
    return split_of


def assign_balanced(sources: list[dict], split_of: dict[str, str], cfg: dict, stream_id: int,
                    frac: float, labels: tuple[str, str]) -> dict[str, str]:
    """Give exactly round(frac * n) sources labels[0] and the rest labels[1] in every
    (split, document type) group; which sources get which label is decided by the seed.
    Used for text-edit difficulty (hard/easy) and splice size class (field/block)."""
    out = {}
    for doc_type, code in DOC_TYPE_CODES.items():
        for si, split in enumerate(CORE_SPLITS):
            ids = [s["source_id"] for s in sources
                   if s["document_type"] == doc_type and split_of[s["source_id"]] == split]
            n_first = int(np.floor(len(ids) * frac + 0.5))
            order = stream(cfg["seed"], stream_id, code, si).permutation(len(ids))
            for rank, j in enumerate(order):
                out[ids[j]] = labels[0] if rank < n_first else labels[1]
    return out


def plan(cfg: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    """Sources annotated with split, text-edit difficulty and splice size class,
    plus the members of each split (the donor pools)."""
    sources = plan_sources(cfg)
    split_of = assign_splits(sources, cfg)
    diff_of = assign_balanced(sources, split_of, cfg, S_DIFFICULTY,
                              cfg["tamper"]["hard_fraction"], ("hard", "easy"))
    cls_of = assign_balanced(sources, split_of, cfg, S_SPLICE_CLASS,
                             cfg["tamper"]["splice"]["field_fraction"], ("field", "block"))
    sources = [dict(s, split=split_of[s["source_id"]], text_difficulty=diff_of[s["source_id"]],
                    splice_size_class=cls_of[s["source_id"]]) for s in sources]
    members = {sp: [s for s in sources if s["split"] == sp] for sp in CORE_SPLITS}
    return sources, members


def render_source(src: dict, cfg: dict, fonts: FontRegistry, families: list[str]):
    """Render the clean page and apply the stage-1 JPEG (if compression.double)."""
    seed = cfg["seed"]
    rng = stream(seed, src["type_code"], src["index"], S_LAYOUT)
    page = RENDERERS[src["document_type"]](rng, cfg["page"], cfg["render"], fonts, families)
    q1 = None
    base = page.image
    if cfg["compression"]["double"]:
        lo, hi = cfg["compression"]["stage1_quality"]
        q1 = int(stream(seed, src["type_code"], src["index"], S_STAGE1).integers(lo, hi + 1))
        base = jpeg_roundtrip(page.image, q1)
    return page, base, q1


def choose_donor(src: dict, members: list[dict], rng, same_type: bool) -> dict:
    """Pick a donor source from the SAME split (`members`), never the source itself."""
    pool = [m for m in members if m["source_id"] != src["source_id"]]
    if same_type:
        same = [m for m in pool if m["document_type"] == src["document_type"]]
        pool = same or pool
    if not pool:
        raise ValueError(f"no donor available in split for {src['source_id']}")
    return pool[int(rng.integers(len(pool)))]


def variant_list(cfg: dict) -> list[tuple[str, int]]:
    return [(v, k) for v in ("authentic", "text_overwrite", "splice") for k in range(int(cfg["variants"].get(v, 0)))]


_FONTS: FontRegistry | None = None


def _fonts(cfg) -> FontRegistry:
    global _FONTS
    if _FONTS is None:
        _FONTS = FontRegistry(cfg["fonts"])
    return _FONTS


def _alternate(label: str, k: int, pair: tuple[str, str]) -> str:
    """Extra variants (k = 1, 3, ...) of the same source get the other label."""
    return label if k % 2 == 0 else (pair[1] if label == pair[0] else pair[0])


def make_variant(src: dict, members: list[dict], cfg: dict, page, base: np.ndarray,
                 vtype: str, k: int, families: list[str]) -> dict:
    """Create one variant (before post-processing) of an already rendered source.

    Returns image, mask (render coordinates) and the metadata pieces; for edits also the
    EditResult and, for text overwrites, the edited TextField (used by tests to rebuild and
    inspect a saved edit exactly)."""
    fonts = _fonts(cfg)
    seed, tc, si = cfg["seed"], src["type_code"], src["index"]
    vrng = stream(seed, tc, si, S_VARIANT, VARIANT_CODES[vtype], k)
    out = {"difficulty": None, "splice_size_class": None, "edit": None, "donor_source_id": None,
           "donor_box_xyxy": None, "result": None, "field": None}
    if vtype == "authentic":
        out.update(image=base.copy(), mask=np.zeros(base.shape[:2], bool))
    elif vtype == "text_overwrite":
        difficulty = _alternate(src["text_difficulty"], k, ("hard", "easy"))
        fields = page.editable_fields()
        field = fields[int(vrng.integers(len(fields)))]
        res = text_overwrite(base, field, difficulty, cfg["tamper"], fonts, families, vrng)
        out.update(image=res.image, mask=res.mask, difficulty=difficulty, edit=res.info,
                   result=res, field=field)
    else:
        size_class = _alternate(src["splice_size_class"], k, ("field", "block"))
        donor = choose_donor(src, members, stream(seed, tc, si, S_DONOR, k),
                             cfg["tamper"]["splice"]["donor_same_type"])
        donor_page, donor_img, donor_q1 = render_source(donor, cfg, fonts, families)
        res = splice(base, donor_img, donor_page.fields, size_class, cfg["tamper"]["splice"], vrng,
                     cfg["crops"]["ink_gray_threshold"])
        res.info["donor_stage1_jpeg_quality"] = donor_q1
        out.update(image=res.image, mask=res.mask, splice_size_class=size_class, edit=res.info,
                   donor_source_id=donor["source_id"], donor_box_xyxy=res.info["donor_box_xyxy"],
                   result=res)
    return out


def generate_source(src: dict, members: list[dict], cfg: dict, data_root: Path,
                    condition: str = "standard", families: list[str] | None = None,
                    write: bool = True) -> list[dict]:
    """Generate all variants of one source. Returns metadata dicts (and writes files).

    With write=False nothing is saved; each dict then also carries the decoded `_image`
    and `_mask` arrays (used by the determinism test)."""
    cv2.setNumThreads(1)
    families = families or cfg["fonts"]["core"]
    seed, tc, si, split = cfg["seed"], src["type_code"], src["index"], src["split"]
    page, base, q1 = render_source(src, cfg, _fonts(cfg), families)
    if write:
        (data_root / "generated" / split).mkdir(parents=True, exist_ok=True)
    records = []
    for vtype, k in variant_list(cfg):
        vcode = VARIANT_CODES[vtype]
        sample_id = f"{src['source_id']}_{vtype}_{k}"
        meta = make_variant(src, members, cfg, page, base, vtype, k, families)
        img, mask = meta["image"], meta["mask"]
        post = sample_post_params(stream(seed, tc, si, S_POST, vcode, k), cfg)
        img2, mask2 = apply_post(img, mask, post)
        jpg = encode_jpeg(img2, post["final_jpeg_quality"])
        ys, xs = np.nonzero(mask2)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
        rel = f"generated/{split}/{sample_id}"
        rec = {
            "sample_id": sample_id, "source_id": src["source_id"], "document_type": src["document_type"],
            "split": split, "condition": condition,
            "image_path": rel + ".jpg", "mask_path": rel + "_mask.png", "meta_path": rel + ".json",
            "image_label": int(vtype != "authentic"),
            "manipulation_type": "none" if vtype == "authentic" else vtype,
            "difficulty": meta["difficulty"], "splice_size_class": meta["splice_size_class"],
            "bbox_xyxy": bbox,
            "mask_positive_pixels": int(mask2.sum()),
            "seed": int(seed), "seed_key": {"type_code": tc, "index": si, "variant_code": vcode, "k": k},
            "generator_version": cfg["generator_version"],
            "font_family": page.body_family, "header_font_family": page.header_family,
            "layout": page.layout,
            "render_size_wh": [int(page.image.shape[1]), int(page.image.shape[0])],
            "image_size_wh": [int(img2.shape[1]), int(img2.shape[0])],
            "processing": {"compression_double": bool(cfg["compression"]["double"]),
                           "stage1_jpeg_quality": q1, **post},
            "donor_source_id": meta["donor_source_id"], "donor_box_xyxy": meta["donor_box_xyxy"],
            "edit": meta["edit"],
            "software": software_versions(),
        }
        if write:
            (data_root / rec["image_path"]).write_bytes(jpg)
            Image.fromarray(mask2.astype(np.uint8) * 255, mode="L").save(data_root / rec["mask_path"])
            (data_root / rec["meta_path"]).write_text(json.dumps(rec, indent=2))
        else:
            rec["_image"], rec["_mask"] = decode_image(jpg), mask2
        records.append(rec)
    return records


def _work(args):
    return generate_source(*args)


def build_dataset(cfg: dict, data_root: Path, workers: int | None = None) -> dict[str, list[dict]]:
    """Generate every core split and write data_root/manifests/{split}.jsonl."""
    data_root = Path(data_root)
    sources, members = plan(cfg)
    jobs = [(src, members[src["split"]], cfg, data_root) for src in sources]
    workers = cfg["build"]["workers"] if workers is None else workers
    if workers and workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_work, jobs, chunksize=2))
    else:
        results = [_work(j) for j in jobs]
    by_split = {s: [] for s in CORE_SPLITS}
    for recs in results:
        for r in recs:
            by_split[r["split"]].append(r)
    man_dir = data_root / "manifests"
    man_dir.mkdir(parents=True, exist_ok=True)
    for split, recs in by_split.items():
        recs.sort(key=lambda r: r["sample_id"])
        with open(man_dir / f"{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, sort_keys=True) + "\n")
    info = {"generator_version": cfg["generator_version"], "seed": cfg["seed"],
            "software": software_versions(), "config": cfg,
            "counts": {s: len(r) for s, r in by_split.items()},
            "sources": {s: len(m) for s, m in members.items()}}
    (man_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    return by_split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--data-root", default=None, help="override paths.data_root")
    ap.add_argument("--workers", type=int, default=None, help="override build.workers")
    ap.add_argument("--no-report", action="store_true", help="skip hash, counts and contact sheet")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_root = resolve(args.data_root or cfg["paths"]["data_root"])
    workers = args.workers if args.workers is not None else min(cfg["build"]["workers"], os.cpu_count() or 1)
    by_split = build_dataset(cfg, data_root, workers)
    print("generated:", {s: len(r) for s, r in by_split.items()}, "->", data_root)
    if not args.no_report:
        from research.gen import dataset_hash, report
        info = dataset_hash.compute_dataset_hash(data_root, CORE_SPLITS)
        dataset_hash.write_hash_file(resolve(cfg["paths"]["hash_file"]), info)
        print("dataset sha256:", info["sha256"], "->", cfg["paths"]["hash_file"])
        report.write_counts(by_split, resolve(cfg["paths"]["counts_csv"]))
        report.contact_sheet(by_split, data_root, resolve(cfg["paths"]["contact_sheet"]), seed=cfg["seed"])
        print("contact sheet ->", cfg["paths"]["contact_sheet"])


if __name__ == "__main__":
    main()
