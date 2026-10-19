"""U-Net (ResNet-18 encoder) construction, checkpoint I/O and full-page prediction.

Shared by research/train.py, research/evaluate.py and the serving API, so preprocessing
(ImageNet normalisation, size cap, tiling, sigmoid) is identical everywhere.

Checkpoint format (torch.save dict): {"model_state", "arch", "epoch", "val_metric",
"train_config", ...}. Loading never needs the network: the model is built with
encoder_weights=None and the trained weights are loaded on top.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from docforge.preprocess import cap_long_side, imagenet_normalize, resize_map
from docforge.tiling import tiled_predict

ARCH = {"name": "Unet", "encoder": "resnet18", "in_channels": 3, "classes": 1}


def build_model(encoder_weights: str | None = None, weights_path: str | Path | None = None) -> torch.nn.Module:
    """smp.Unet("resnet18", in_channels=3, classes=1).

    encoder_weights: "imagenet" (downloads pretrained weights; needs internet) or None.
    weights_path: optional local file with ImageNet ResNet-18 weights in torchvision format
        (e.g. resnet18-f37072fd.pth uploaded as a Kaggle dataset); loaded into the encoder.
        This is the offline fallback for encoder_weights="imagenet".
    """
    import segmentation_models_pytorch as smp

    model = smp.Unet(ARCH["encoder"], encoder_weights=encoder_weights,
                     in_channels=ARCH["in_channels"], classes=ARCH["classes"])
    if weights_path:
        load_encoder_weights(model, weights_path)
    return model


def load_encoder_weights(model: torch.nn.Module, path: str | Path) -> dict:
    """Load a torchvision-format ResNet-18 state dict into model.encoder (fc.* ignored).
    Raises if any encoder parameter is missing from the file."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in state.items() if not k.startswith("fc.")}
    # call nn.Module.load_state_dict directly: smp's encoder override returns None,
    # and we need the missing/unexpected key lists to detect a wrong weights file
    missing, unexpected = torch.nn.Module.load_state_dict(model.encoder, state, strict=False)
    if missing:
        raise ValueError(f"encoder weights file is missing keys: {missing[:5]}...")
    return {"loaded": len(state), "unexpected": list(unexpected)}


def save_checkpoint(path: str | Path, model: torch.nn.Module, **extra) -> None:
    torch.save({"model_state": model.state_dict(), "arch": ARCH, **extra}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict]:
    """Build the architecture without downloading anything and load trained weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if ckpt.get("arch") != ARCH:
        raise ValueError(f"checkpoint architecture {ckpt.get('arch')} != expected {ARCH}")
    model = build_model(encoder_weights=None)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {k: v for k, v in ckpt.items() if k not in ("model_state", "optimizer_state", "scaler_state",
                                                       "scheduler_state")}
    return model, meta


def make_predict_fn(model: torch.nn.Module, device: str = "cpu", amp: bool = False):
    """Tile predictor for docforge.tiling: uint8 NxTxTx3 -> float32 NxTxT probabilities."""
    model.eval()

    @torch.no_grad()
    def predict(tiles: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(imagenet_normalize(tiles)).to(device)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=amp and device.startswith("cuda")):
            logits = model(x)
        return torch.sigmoid(logits.float())[:, 0].cpu().numpy()

    return predict


def predict_page(img: np.ndarray, predict_fn, max_side: int = 2000, tile: int = 256, stride: int = 128,
                 batch_size: int = 16, floor: float = 0.01) -> np.ndarray:
    """Full-page probability map at the ORIGINAL image size: cap the long side, tiled
    inference with Hann blending, then bilinear resize back."""
    small, _ = cap_long_side(img, max_side)
    prob = tiled_predict(small, predict_fn, tile=tile, stride=stride, batch_size=batch_size, floor=floor)
    return np.clip(resize_map(prob, img.shape[:2]), 0.0, 1.0)
