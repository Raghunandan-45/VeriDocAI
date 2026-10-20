"""CPU latency of the serving API per page, at several tiling strides.

    .venv/bin/python -m research.latency                      # strides 128 192 256, 20 val pages
    .venv/bin/python -m research.latency --strides 128 256 --pages 10

Each page is sent as its original JPEG bytes to POST /v1/analyze through an in-process
TestClient, so the time includes upload parsing, validation, decoding, tiled inference,
region extraction and image encoding, but no network. Pages are the first N validation
pages by sample_id (latency only: no labels are used and no threshold is chosen). Two
warm-up requests per stride are excluded. Run it on an otherwise idle machine.
Writes results/latency_cpu.json.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np
from fastapi.testclient import TestClient

from research.config import resolve
from research.evaluate import read_manifest
from serving.app import create_app, load_settings


def hardware() -> dict:
    import torch

    cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
    mem = next((int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemTotal")), 0)
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "ram_gib": round(mem / 1024 ** 2, 1), "os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "device": "cpu"}


def pct(v, q) -> float:
    return round(float(np.percentile(v, q)), 1)


def summary(v: list[float]) -> dict:
    return {"p50": pct(v, 50), "p95": pct(v, 95), "mean": round(float(np.mean(v)), 1),
            "min": round(float(np.min(v)), 1), "max": round(float(np.max(v)), 1)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--strides", type=int, nargs="+", default=[128, 192, 256])
    ap.add_argument("--pages", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default="results/latency_cpu.json")
    a = ap.parse_args(argv)

    recs = sorted(read_manifest(resolve("data/manifests/val.jsonl")), key=lambda r: r["sample_id"])
    step = max(1, len(recs) // a.pages)
    recs = recs[::step][: a.pages]                      # spread over invoices and marksheets
    blobs = [(r["sample_id"], (resolve("data") / r["image_path"]).read_bytes()) for r in recs]
    out = {"evidence_class": "measured on this machine; in-process HTTP requests (no network)",
           "hardware": hardware(), "n_pages": len(blobs), "warmup_requests": a.warmup,
           "pages": [s for s, _ in blobs], "results": {}}
    for stride in a.strides:
        settings = load_settings()
        settings["inference"]["stride"] = stride
        with TestClient(create_app(settings)) as c:
            assert c.get("/health").status_code == 200, "model not loaded"
            for _, b in blobs[: a.warmup]:
                c.post("/v1/analyze", files={"file": ("p.jpg", b, "image/jpeg")})
            total, infer, sizes = [], [], []
            for sid, b in blobs:
                t0 = time.perf_counter()
                r = c.post("/v1/analyze", files={"file": ("p.jpg", b, "image/jpeg")})
                total.append((time.perf_counter() - t0) * 1e3)
                assert r.status_code == 200, r.text
                d = r.json()
                infer.append(d["timing_ms"]["inference"])
                sizes.append((d["image"]["width"], d["image"]["height"]))
        out["results"][str(stride)] = {"stride": stride, "tile": 256, "request_ms": summary(total),
                                       "inference_ms": summary(infer)}
        print(stride, out["results"][str(stride)], flush=True)
    w, h = np.array(sizes).T
    out["resolution"] = {"width_min": int(w.min()), "width_max": int(w.max()), "height_min": int(h.min()),
                         "height_max": int(h.max()), "median_megapixels": round(float(np.median(w * h)) / 1e6, 2),
                         "analysis_max_side": 2000,
                         "n_pages_downscaled_to_cap": int((np.maximum(w, h) > 2000).sum())}
    path = resolve(a.out)
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
