"""Dataset tests 1-6 and 10 (see README "Dataset tests")."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from research.config import with_overrides
from research.gen.build_dataset import generate_source, make_variant, plan, plan_sources, render_source, stream
from research.gen.fonts import FontRegistry
from research.gen.postprocess import apply_post, decode_image, encode_jpeg, resize_pair
from research.gen.tamper_ops import clean_rerender, text_overwrite
from research.sampler import CropSampler


def all_records(manifests):
    return [r for recs in manifests.values() for r in recs]


# 1 ------------------------------------------------------------------------------------------
def test_one_image_mask_metadata_per_sample(dataset):
    root, manifests, _ = dataset
    recs = all_records(manifests)
    ids = [r["sample_id"] for r in recs]
    assert len(ids) == len(set(ids)), "duplicate sample ids"
    for r in recs:
        for key in ("image_path", "mask_path", "meta_path"):
            assert (root / r[key]).is_file(), f"missing {r[key]}"
    files = list((root / "generated").rglob("*"))
    jpgs = [f for f in files if f.suffix == ".jpg"]
    pngs = [f for f in files if f.name.endswith("_mask.png")]
    jsons = [f for f in files if f.suffix == ".json"]
    assert len(jpgs) == len(pngs) == len(jsons) == len(recs), "orphan or missing files"


# 2, 3, 4 ------------------------------------------------------------------------------------
def test_image_mask_dims_and_mask_content(dataset):
    root, manifests, _ = dataset
    for r in all_records(manifests):
        img = Image.open(root / r["image_path"])
        mask = np.asarray(Image.open(root / r["mask_path"]))
        assert img.mode == "RGB"
        assert mask.ndim == 2, "mask must be single-channel"
        assert (img.size[1], img.size[0]) == mask.shape, f"{r['sample_id']}: image/mask size differ"
        assert list(img.size) == r["image_size_wh"]
        assert set(np.unique(mask)) <= {0, 255}, "mask must be binary 0/255"
        n_pos = int((mask > 0).sum())
        assert n_pos == r["mask_positive_pixels"]
        if r["image_label"] == 0:
            assert n_pos == 0, f"{r['sample_id']}: authentic mask not all black"   # test 3
        else:
            assert n_pos > 0, f"{r['sample_id']}: tampered mask empty"            # test 4
            assert r["bbox_xyxy"] is not None


# 5 ------------------------------------------------------------------------------------------
def _marker(h=500, w=600, box=(213, 141, 287, 199)):
    """Grey page with a pure red rectangle and a mask of exactly that rectangle."""
    img = np.full((h, w, 3), 235, np.uint8)
    img[::7, :, :] = 40  # thin dark lines so the page is not blank
    mask = np.zeros((h, w), bool)
    x0, y0, x1, y1 = box
    img[y0:y1, x0:x1] = (255, 0, 0)
    mask[y0:y1, x0:x1] = True
    return img, mask


def _red(img):
    return (img[..., 0] > 170) & (img[..., 1] < 90) & (img[..., 2] < 90)


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def _centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()])


@pytest.mark.parametrize("scale", [0.8, 0.93, 1.07, 1.2])
@pytest.mark.parametrize("interp", ["bilinear", "lanczos"])
def test_mask_alignment_survives_resize_and_postprocess(scale, interp):
    img, mask = _marker()
    img2, mask2 = resize_pair(img, mask, scale, interp)
    assert img2.shape[:2] == mask2.shape
    assert _iou(_red(img2), mask2) > 0.9
    assert np.abs(_centroid(_red(img2)) - _centroid(mask2)).max() < 1.0
    # full post-processing incl. blur and final JPEG
    params = {"resize_scale": scale, "resize_interp": interp, "blur_applied": True,
              "blur_sigma": 1.0, "final_jpeg_quality": 60}
    img3, mask3 = apply_post(img, mask, params)
    img3 = decode_image(encode_jpeg(img3, 60))
    assert img3.shape[:2] == mask3.shape
    assert _iou(_red(img3), mask3) > 0.85
    assert np.abs(_centroid(_red(img3)) - _centroid(mask3)).max() < 1.0


def test_mask_alignment_survives_sampler_crops(tmp_path, data_cfg):
    img, mask = _marker(700, 800, (301, 255, 377, 331))
    Image.fromarray(img).save(tmp_path / "m.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(tmp_path / "m_mask.png")
    rec = {"sample_id": "marker", "split": "train", "image_label": 1, "image_path": "m.png",
           "mask_path": "m_mask.png", "bbox_xyxy": [301, 255, 377, 331]}
    crops_cfg = dict(data_cfg["crops"], p_positive=1.0)
    crops_cfg["aug"] = {"scale_prob": 0.5, "scale_range": [0.9, 1.1]}
    s = CropSampler([rec], tmp_path, crops_cfg)
    rng = np.random.default_rng(0)
    windows = set()
    for _ in range(40):
        c = s.sample(rng)
        windows.add(c.window)
        red, m = _red(c.image), c.mask.astype(bool)
        assert _iou(red, m) > 0.85, f"crop at {c.xy} win {c.window} misaligned"
    assert len(windows) > 1, "scale jitter never exercised"


# 6 ------------------------------------------------------------------------------------------
def test_same_seed_identical_decoded_pixels(tiny_cfg):
    sources, members = plan(tiny_cfg)
    strip = lambda r: {k: v for k, v in r.items() if not k.startswith("_")}  # noqa: E731
    for src in (sources[0], sources[-1]):  # one invoice, one marksheet
        a = generate_source(src, members[src["split"]], tiny_cfg, Path("."), write=False)
        b = generate_source(src, members[src["split"]], tiny_cfg, Path("."), write=False)
        for ra, rb in zip(a, b):
            assert np.array_equal(ra["_image"], rb["_image"])
            assert np.array_equal(ra["_mask"], rb["_mask"])
            assert strip(ra) == strip(rb)
    # sanity: a different global seed must change the pixels
    other = with_overrides(tiny_cfg, {"seed": tiny_cfg["seed"] + 1})
    o_sources, o_members = plan(other)
    c = generate_source(o_sources[0], o_members[o_sources[0]["split"]], other, Path("."), write=False)
    a = generate_source(sources[0], members[sources[0]["split"]], tiny_cfg, Path("."), write=False)
    assert not np.array_equal(a[0]["_image"], c[0]["_image"])


# 10 -----------------------------------------------------------------------------------------
def test_saved_hard_edits_not_pixel_identical_to_clean_rerender(dataset):
    """For EVERY hard text edit in the dataset: rebuild the edit from its seed, check it matches
    the saved metadata and reproduces the saved image file exactly, then compare it with a
    clean re-render of the same new value (same font, size, colour, position, default
    anti-aliasing): before post-processing at least 10 pixels must differ by >= 3 grey levels,
    and after the sample's own resize/blur/JPEG the saved image must still differ."""
    root, manifests, cfg = dataset
    sources, members = plan(cfg)
    by_id = {s["source_id"]: s for s in sources}
    fonts = FontRegistry(cfg["fonts"])
    fams = cfg["fonts"]["core"]
    hard = [r for recs in manifests.values() for r in recs if r["difficulty"] == "hard"]
    pre, post_changed = [], []
    for r in hard:
        src = by_id[r["source_id"]]
        page, base, _ = render_source(src, cfg, fonts, fams)
        v = make_variant(src, members[src["split"]], cfg, page, base, "text_overwrite", r["seed_key"]["k"], fams)
        assert json.loads(json.dumps(v["edit"])) == r["edit"], f"{r['sample_id']}: rebuilt edit != metadata"
        clean = clean_rerender(v["result"].covered, v["field"], v["edit"]["new_value"], fonts)
        changed = int((np.abs(v["image"].astype(int) - clean.astype(int)).max(axis=2) >= 3).sum())
        assert changed >= 10, f"{r['sample_id']}: only {changed} px differ from a clean re-render"
        p = {k: r["processing"][k] for k in ("resize_scale", "resize_interp", "blur_applied", "blur_sigma")}
        q = r["processing"]["final_jpeg_quality"]
        saved = np.asarray(Image.open(root / r["image_path"]).convert("RGB"))
        final = decode_image(encode_jpeg(apply_post(v["image"], v["mask"], p)[0], q))
        assert np.array_equal(final, saved), f"{r['sample_id']}: rebuild does not reproduce saved image"
        clean_final = decode_image(encode_jpeg(apply_post(clean, v["mask"], p)[0], q))
        n_final = int((np.abs(clean_final.astype(int) - saved.astype(int)).max(axis=2) >= 3).sum())
        assert n_final > 0, f"{r['sample_id']}: identical to clean re-render after post-processing"
        pre.append(changed)
        post_changed.append(n_final)
    assert len(hard) > 0
    print(f"{len(hard)} saved hard edits; px differing >=3 levels: before post min {min(pre)} "
          f"median {int(np.median(pre))}; after post min {min(post_changed)} median {int(np.median(post_changed))}")


def test_forced_hard_edits_on_many_fields(tiny_cfg):
    """Stress version of test 10: force a hard edit on 6 fields of every other tiny source."""
    fonts = FontRegistry(tiny_cfg["fonts"])
    families = tiny_cfg["fonts"]["core"]
    n_checked, min_changed = 0, None
    for src in plan_sources(tiny_cfg)[::2]:
        page, base, _ = render_source(src, tiny_cfg, fonts, families)
        for j, field in enumerate(page.editable_fields()[:6]):
            rng = stream(999, src["type_code"], src["index"], j)
            res = text_overwrite(base, field, "hard", tiny_cfg["tamper"], fonts, families, rng)
            clean = clean_rerender(res.covered, field, res.info["new_value"], fonts)
            diff = np.abs(res.image.astype(int) - clean.astype(int)).max(axis=2)
            changed = int((diff >= 3).sum())
            assert changed >= 10, (f"{src['source_id']} {field.name}: hard edit nearly identical to clean "
                                   f"re-render ({changed} px differ); info={res.info}")
            assert res.info["edit_font_family"] == field.family
            assert abs(res.info["edit_font_size"] - field.size) <= 1
            min_changed = changed if min_changed is None else min(min_changed, changed)
            n_checked += 1
    assert n_checked >= 50
    print(f"checked {n_checked} hard edits; min pixels differing (>=3 levels) = {min_changed}")
