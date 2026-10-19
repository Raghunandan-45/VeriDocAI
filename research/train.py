"""Train the U-Net (ResNet-18 encoder) on 256x256 crops from the TRAIN split.

    # tiny overfit check (CPU is fine): loss must decrease
    .venv/bin/python -m research.train --smoke --allow-cpu --encoder-weights none
    # full training (GPU, e.g. Kaggle T4)
    python -m research.train --data-root /kaggle/input/<dataset>/data --output-dir /kaggle/working/run
    # continue after a disconnect
    python -m research.train --data-root ... --output-dir /kaggle/working/run --resume

Each DataLoader item is one PAGE VISIT: one page is decoded and `crops_per_visit` crops are
cut from it (research/sampler.py), so JPEG decoding is amortised over several crops.
Item i of epoch e uses rng([seed, e, i]), so crops do not depend on worker count.

Per epoch: train, validate on a fixed subset of validation pages (full pages, coarser
stride), append a row to metrics.csv, save last.pt (with optimizer state for --resume) and
best.pt (best validation-subset F1). At the end the best checkpoint is validated on all
validation pages at the final stride (final_val.json). These validation numbers are used for
checkpoint selection and are therefore optimistic; frozen thresholds and the held-out test
evaluation are produced separately by research/evaluate.py.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

from research.config import load_config, resolve
from research.gen.dataset_hash import read_manifest
from research.sampler import CropSampler
from docforge.model import ARCH, build_model, make_predict_fn, predict_page, save_checkpoint
from docforge.preprocess import imagenet_normalize, load_mask, load_rgb
from docforge.scoring import f1_curve, pixel_hists, roc_auc


# --- data -------------------------------------------------------------------------------------
class VisitDataset(Dataset):
    """Item i -> (images k x 3 x 256 x 256 float32, masks k x 1 x 256 x 256 float32)."""

    def __init__(self, records, data_root, crops_cfg, n_visits, crops_per_visit, seed, page_cache):
        self.records, self.data_root, self.crops_cfg = records, Path(data_root), crops_cfg
        self.n_visits, self.k, self.seed, self.page_cache = n_visits, crops_per_visit, seed, page_cache
        self.epoch = 0
        self.sampler = None  # created lazily inside each worker process

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.n_visits

    def __getitem__(self, i):
        if self.sampler is None:
            self.sampler = CropSampler(self.records, self.data_root, self.crops_cfg, self.page_cache)
        crops = self.sampler.sample_visit(np.random.default_rng([self.seed, self.epoch, i]), self.k)
        img = np.stack([c.image for c in crops])
        mask = np.stack([c.mask for c in crops])[:, None].astype(np.float32)
        return torch.from_numpy(imagenet_normalize(img)), torch.from_numpy(mask)


def collate_visits(batch):
    return torch.cat([b[0] for b in batch]), torch.cat([b[1] for b in batch])


def val_subset(records: list[dict], n: int, seed: int) -> list[dict]:
    """Fixed subset: n/3 authentic + 2n/3 tampered validation pages, chosen by seed."""
    rng = np.random.default_rng([seed, 4242])
    auth = sorted([r for r in records if r["image_label"] == 0], key=lambda r: r["sample_id"])
    tam = sorted([r for r in records if r["image_label"] == 1], key=lambda r: r["sample_id"])
    na = min(len(auth), n // 3)
    nt = min(len(tam), n - na)
    pick = [auth[i] for i in rng.choice(len(auth), na, replace=False)] + \
           [tam[i] for i in rng.choice(len(tam), nt, replace=False)]
    return sorted(pick, key=lambda r: r["sample_id"])


# --- loss / validation ------------------------------------------------------------------------
def bce_dice_loss(logits, target, bce_w=1.0, dice_w=1.0, eps=1.0):
    """BCE-with-logits + soft Dice (computed over the whole batch). Returns (total, bce, dice)."""
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1.0 - (2.0 * (p * target).sum() + eps) / (p.sum() + target.sum() + eps)
    return bce_w * bce + dice_w * dice, bce, dice


def validate(model, records, data_root, stride, eval_cfg, n_bins, device, amp) -> dict:
    """Full-page validation: best pooled pixel F1 over all thresholds (threshold-free model
    selection metric) and image ROC-AUC of the max-probability score."""
    predict_fn = make_predict_fn(model, device, amp)
    t = eval_cfg["tiling"]
    pos, neg, smax, labels = [], [], [], []
    for r in records:
        img = load_rgb(Path(data_root) / r["image_path"])
        gt = load_mask(Path(data_root) / r["mask_path"])
        prob = predict_page(img, predict_fn, eval_cfg["analysis"]["max_side"], t["tile"], stride,
                            t["batch_size"], t["window_floor"])
        p, n = pixel_hists(prob, gt, n_bins)
        pos.append(p)
        neg.append(n)
        smax.append(float(prob.max()))
        labels.append(int(r["image_label"]))
    curve = f1_curve(np.sum(pos, 0), np.sum(neg, 0))
    k = int(np.argmax(curve))
    return {"best_pooled_f1": float(curve[k]), "best_threshold": k / n_bins,
            "auc_max": roc_auc(labels, smax), "n_pages": len(records), "stride": stride}


# --- helpers -----------------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sums(out_dir: Path) -> None:
    """Print sha256 and size of every output file and write SHA256SUMS.txt."""
    lines = []
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            digest = sha256_file(p)
            lines.append(f"{digest}  {p.name}")
            print(f"{p.name:24s} {p.stat().st_size:>12,d} bytes  sha256 {digest}")
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")


def grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def environment(device: str) -> dict:
    env = {"python": platform.python_version(), "torch": str(torch.__version__), "device": device,
           "cuda_available": torch.cuda.is_available()}
    if device.startswith("cuda"):
        env["gpu"] = torch.cuda.get_device_name(0)
    return env


def encoder_weights_arg(value):
    return None if value in (None, "none", "None", "null") else value


# --- smoke test ------------------------------------------------------------------------------
def smoke(cfg, crops_cfg, data_root, out_dir, device, amp, encoder_weights) -> dict:
    """Overfit a tiny fixed batch; the loss must go down."""
    sc = cfg["smoke"]
    train = read_manifest(data_root / "manifests" / "train.jsonl")
    pages = sorted([r for r in train if r["image_label"] == 1], key=lambda r: r["sample_id"])[: sc["pages"]]
    pages += sorted([r for r in train if r["image_label"] == 0], key=lambda r: r["sample_id"])[: sc["pages"]]
    sampler = CropSampler(pages, data_root, crops_cfg, cache_pages=2 * sc["pages"])
    rng = np.random.default_rng([cfg["seed"], 777])
    crops = [sampler.sample(rng) for _ in range(sc["crops"])]
    x = torch.from_numpy(imagenet_normalize(np.stack([c.image for c in crops]))).to(device)
    y = torch.from_numpy(np.stack([c.mask for c in crops])[:, None].astype(np.float32)).to(device)
    torch.manual_seed(cfg["seed"])
    model = build_model(encoder_weights, cfg["model"]["weights_path"]).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=sc["lr"])
    scaler = grad_scaler(amp)
    losses = []
    for step in range(sc["steps"]):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
            loss, bce, dice = bce_dice_loss(model(x), y, cfg["train"]["bce_weight"], cfg["train"]["dice_weight"])
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        print(f"smoke step {step + 1:2d}/{sc['steps']}  loss {losses[-1]:.4f}  (bce {bce.item():.4f}, dice {dice.item():.4f})")
    first, last = float(np.mean(losses[:5])), float(np.mean(losses[-5:]))
    res = {"device": device, "amp": amp, "encoder_weights": encoder_weights, "n_crops": len(crops),
           "n_positive_crops": int(sum(c.is_positive for c in crops)), "steps": sc["steps"], "lr": sc["lr"],
           "losses": [round(v, 5) for v in losses], "mean_first5": first, "mean_last5": last,
           "decreased": last < first}
    (out_dir / "smoke_losses.json").write_text(json.dumps(res, indent=2))
    print(f"smoke: mean loss first 5 steps {first:.4f} -> last 5 steps {last:.4f}: "
          f"{'DECREASED (PASS)' if res['decreased'] else 'DID NOT DECREASE (FAIL)'}")
    return res


# --- main --------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--amp", dest="amp", action="store_true", default=None, help="mixed precision on (GPU only)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", help="mixed precision off")
    ap.add_argument("--encoder-weights", default=None, help="imagenet | none (overrides config)")
    ap.add_argument("--weights-path", default=None, help="offline torchvision resnet18 weights file")
    ap.add_argument("--resume", action="store_true", help="continue from <output-dir>/last.pt")
    ap.add_argument("--smoke", action="store_true", help="tiny overfit check only")
    ap.add_argument("--allow-cpu", action="store_true", help="permit running without a GPU")
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="debug: cap steps per epoch")
    ap.add_argument("--val-pages", type=int, default=None, help="debug: override val.subset_pages")
    ap.add_argument("--skip-final-val", action="store_true", help="debug: skip full validation at the end")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    for key, val in [("batch_size", a.batch_size), ("epochs", a.epochs), ("lr", a.lr), ("num_workers", a.num_workers),
                     ("amp", a.amp)]:
        if val is not None:
            cfg["train"][key] = val
    if a.encoder_weights is not None:
        cfg["model"]["encoder_weights"] = a.encoder_weights
    if a.weights_path is not None:
        cfg["model"]["weights_path"] = a.weights_path
    if a.val_pages is not None:
        cfg["val"]["subset_pages"] = a.val_pages
    data_root = resolve(a.data_root or cfg["data_root"])
    out_dir = resolve(a.output_dir or cfg["output_dir"])
    cfg["data_root"], cfg["output_dir"] = str(data_root), str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not a.allow_cpu:
        sys.exit("No CUDA GPU available. Training is meant for a GPU (Kaggle T4); pass --allow-cpu to run on CPU anyway.")
    amp = bool(cfg["train"]["amp"]) and device == "cuda"
    enc_w = encoder_weights_arg(cfg["model"]["encoder_weights"])
    crops_cfg = load_config(cfg["data_config"])["crops"]
    eval_cfg = load_config(cfg["eval_config"])
    if not (data_root / "manifests" / "train.jsonl").exists():
        sys.exit(f"no manifests under {data_root}/manifests (wrong --data-root?)")
    print("environment:", environment(device), "| amp:", amp, "| data_root:", data_root)

    if a.smoke:
        res = smoke(cfg, crops_cfg, data_root, out_dir, device, amp, enc_w)
        sys.exit(0 if res["decreased"] else 1)

    tr = cfg["train"]
    k = int(tr["crops_per_visit"])
    if tr["batch_size"] % k:
        sys.exit(f"batch_size {tr['batch_size']} must be divisible by crops_per_visit {k}")
    visits_per_batch = tr["batch_size"] // k
    train_recs = read_manifest(data_root / "manifests" / "train.jsonl")
    val_recs = read_manifest(data_root / "manifests" / "val.jsonl")
    subset = val_subset(val_recs, cfg["val"]["subset_pages"], cfg["seed"])
    ds = VisitDataset(train_recs, data_root, crops_cfg, tr["visits_per_epoch"], k, cfg["seed"], tr["page_cache"])
    steps_per_epoch = len(ds) // visits_per_batch
    if a.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, a.max_steps_per_epoch)

    torch.manual_seed(cfg["seed"])
    model = build_model(enc_w, cfg["model"]["weights_path"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    total_steps = steps_per_epoch * tr["epochs"]
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total_steps, 1))
             if tr["scheduler"] == "cosine" else torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0))
    scaler = grad_scaler(amp)
    start_epoch, best = 0, -1.0
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if a.resume:
        if not last_path.exists():
            sys.exit(f"--resume given but {last_path} does not exist")
        ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        sched.load_state_dict(ck["scheduler_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_epoch, best = ck["epoch"] + 1, ck["best_val_metric"]
        if isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR):
            sched.T_max = max(total_steps, 1)  # keep the schedule length if --epochs changed
        print(f"resumed from epoch {ck['epoch'] + 1} (best val-subset F1 so far {best:.4f})")

    used = {"train_config": cfg, "crops": crops_cfg, "tiling": eval_cfg["tiling"], "analysis": eval_cfg["analysis"],
            "environment": environment(device), "amp": amp, "steps_per_epoch": steps_per_epoch,
            "val_subset_ids": [r["sample_id"] for r in subset], "argv": sys.argv[1:] if argv is None else argv}
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump(used, sort_keys=False))
    shutil.copy(resolve(cfg["data_config"]), out_dir / "data_config.yaml")
    metrics_path = out_dir / "metrics.csv"
    fields = ["epoch", "steps", "train_loss", "train_bce", "train_dice", "val_subset_best_f1",
              "val_subset_best_threshold", "val_subset_auc_max", "lr_end", "epoch_seconds"]
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    for epoch in range(start_epoch, tr["epochs"]):
        t0 = time.time()
        ds.set_epoch(epoch)
        loader = DataLoader(ds, batch_size=visits_per_batch, shuffle=False, num_workers=tr["num_workers"],
                            collate_fn=collate_visits, drop_last=True, pin_memory=device == "cuda")
        model.train()
        sums, n = np.zeros(3), 0
        for step, (x, y) in enumerate(loader):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp):
                loss, bce, dice = bce_dice_loss(model(x), y, tr["bce_weight"], tr["dice_weight"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            sums += [loss.item(), bce.item(), dice.item()]
            n += 1
            if (step + 1) % tr["log_every"] == 0 or step == 0:
                print(f"epoch {epoch + 1} step {step + 1}/{steps_per_epoch} loss {loss.item():.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        v = validate(model, subset, data_root, cfg["val"]["subset_stride"], eval_cfg, cfg["val"]["n_bins"], device, amp)
        row = {"epoch": epoch + 1, "steps": n, "train_loss": float(sums[0] / max(n, 1)),
               "train_bce": float(sums[1] / max(n, 1)), "train_dice": float(sums[2] / max(n, 1)), "val_subset_best_f1": v["best_pooled_f1"],
               "val_subset_best_threshold": v["best_threshold"], "val_subset_auc_max": v["auc_max"],
               "lr_end": sched.get_last_lr()[0], "epoch_seconds": round(time.time() - t0, 1)}
        with open(metrics_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print("epoch summary:", {k2: (round(v2, 5) if isinstance(v2, float) else v2) for k2, v2 in row.items()},
              flush=True)
        improved = v["best_pooled_f1"] > best
        best = max(best, v["best_pooled_f1"])
        torch.save({"model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                    "scheduler_state": sched.state_dict(), "scaler_state": scaler.state_dict(),
                    "epoch": epoch, "best_val_metric": best, "arch": ARCH, "train_config": cfg}, last_path)
        if improved:
            save_checkpoint(best_path, model, epoch=epoch + 1, val_metric="val_subset_best_pooled_f1",
                            val_value=v["best_pooled_f1"], val_subset_threshold=v["best_threshold"],
                            train_config=cfg)
            print(f"new best checkpoint (val-subset F1 {best:.4f}) -> {best_path}")

    if not a.skip_final_val and best_path.exists():
        ck = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model_state"])
        v = validate(model, val_recs, data_root, cfg["val"]["final_stride"], eval_cfg, cfg["val"]["n_bins"],
                     device, amp)
        v["note"] = ("best checkpoint on ALL validation pages; used for checkpoint selection, so optimistic. "
                     "Frozen thresholds and test results come from research/evaluate.py.")
        v["best_epoch"] = ck["epoch"]
        (out_dir / "final_val.json").write_text(json.dumps(v, indent=2))
        print("final full validation:", v)
    write_sums(out_dir)


if __name__ == "__main__":
    main()
