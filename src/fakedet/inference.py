"""Running models on manifests: logits, embeddings and (cached) frozen-CLIP features."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from fakedet.config import Config
from fakedet.data.dataset import make_loader, to_float
from fakedet.models import clip_layers_needed
from fakedet.models.base import Detector
from fakedet.utils import ensure_dir


def autocast_ctx(device: torch.device, enabled: bool = True):
    on = enabled and device.type == "cuda"
    dtype = torch.bfloat16 if on and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=on)


def _loader(cfg: Config, manifest: pd.DataFrame, batch_size: int, jpeg_quality: int | None = None):
    return make_loader(manifest, cfg.project.paths["data_dir"], cfg.data.crop_size, batch_size, train=False,
                       num_workers=cfg.project.num_workers, jpeg_quality=jpeg_quality)


# ----------------------------------------------------------------- frozen CLIP features
def _feature_key(model: Detector, manifest: pd.DataFrame, crop: int, jpeg_quality: int | None) -> str:
    h = hashlib.sha1()
    h.update(f"{model.backbone_id}|{crop}|{jpeg_quality}".encode())
    h.update("\n".join(manifest.path).encode())
    return h.hexdigest()[:20]


@torch.no_grad()
def clip_features(model: Detector, manifest: pd.DataFrame, cfg: Config, device: torch.device,
                  desc: str = "", jpeg_quality: int | None = None) -> dict:
    """{layer: float16 tensor (N, D)} for every CLIP layer used by the configured variants.
    Computed once per image list and cached on disk, since the backbone is frozen."""
    layers = clip_layers_needed(cfg)
    if model.layer not in layers:
        layers.append(model.layer)
    cache = ensure_dir(Path(cfg.project.paths["cache_dir"]) / "clip_features") / \
        f"{_feature_key(model, manifest, cfg.data.crop_size, jpeg_quality)}.pt"
    if cache.exists():
        feats = torch.load(cache)
        if all(l in feats for l in layers):
            return feats
    model.to(device).eval()
    out = {l: [] for l in layers}
    for x, _ in tqdm(_loader(cfg, manifest, cfg.eval.batch_size, jpeg_quality), desc=f"CLIP features {desc}",
                     leave=False):
        with autocast_ctx(device):
            f = model.extract(to_float(x.to(device)), layers)
        for l in layers:
            out[l].append(f[l].float().cpu())
    feats = {l: torch.cat(v).half() for l, v in out.items()}
    torch.save(feats, cache)
    return feats


# ----------------------------------------------------------------- generic inference
@torch.no_grad()
def predict_logits(model: Detector, manifest: pd.DataFrame, cfg: Config, device: torch.device,
                   jpeg_quality: int | None = None, desc: str = "") -> np.ndarray:
    model.to(device).eval()
    if model.frozen_backbone:  # head on cached backbone features
        feats = clip_features(model, manifest, cfg, device, desc, jpeg_quality)[model.layer]
        out = [model.head(feats[i:i + 4096].float().to(device)).squeeze(1).float().cpu()
               for i in range(0, len(feats), 4096)]
        return torch.cat(out).numpy()
    out = []
    for x, _ in tqdm(_loader(cfg, manifest, cfg.eval.batch_size, jpeg_quality), desc=f"predict {desc}", leave=False):
        with autocast_ctx(device, enabled=model.frozen_backbone):
            out.append(model(to_float(x.to(device))).float().cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def extract_embeddings(model: Detector, manifest: pd.DataFrame, cfg: Config, device: torch.device,
                       desc: str = "") -> np.ndarray:
    """Pre-logit features (N, D) of the detector for every image of the manifest."""
    model.to(device).eval()
    if model.frozen_backbone:
        return clip_features(model, manifest, cfg, device, desc)[model.layer].float().numpy()
    out = []
    for x, _ in tqdm(_loader(cfg, manifest, cfg.eval.batch_size), desc=f"embed {desc}", leave=False):
        out.append(model.embed(to_float(x.to(device))).float().cpu())
    return torch.cat(out).numpy()
