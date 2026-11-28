"""Image loading: crop-never-resize preprocessing and the torch Dataset / DataLoader over manifest rows.

Images are CROPPED (random for training, centre for evaluation) so resampling never smooths away a generator's
high-frequency fingerprint. The only exception is an image smaller than the crop (BigGAN, 128 px), which is
upsampled until it covers the crop.  Samples are uint8 tensors; models normalise internally (`to_float`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fakedet.data.formats import recompress


def fit_and_crop(img: Image.Image, size: int, random_crop: bool = False) -> Image.Image:
    """`size` x `size` crop of `img`; upsamples first only if the image is smaller than the crop."""
    w, h = img.size
    if min(w, h) < size:
        scale = size / min(w, h)
        img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.BICUBIC)
        w, h = img.size
    if random_crop:  # torch RNG: seeded per DataLoader worker, so runs are reproducible
        left = int(torch.randint(0, w - size + 1, (1,)))
        top = int(torch.randint(0, h - size + 1, (1,)))
    else:
        left, top = (w - size) // 2, (h - size) // 2
    return img.crop((left, top, left + size, top + size))


class ImageDataset(Dataset):
    """Rows of a manifest -> (uint8 tensor (3, S, S), float label 0 = real / 1 = fake).

    train=True augments with `augment` ({hflip, random_crop}, both on by default); otherwise a plain centre crop.
    jpeg_quality re-encodes the crop as JPEG (test-time compression robustness).
    """

    def __init__(self, manifest: pd.DataFrame, data_dir: str | Path, crop_size: int, train: bool = False,
                 augment: dict | None = None, jpeg_quality: int | None = None):
        self.paths = manifest.path.tolist()
        self.labels = manifest.label.to_numpy(dtype=np.float32)
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        augment = augment if augment is not None else {"hflip": True, "random_crop": True}
        self.hflip = train and bool(augment.get("hflip", False))
        self.random_crop = train and bool(augment.get("random_crop", False))
        self.jpeg_quality = jpeg_quality

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        with Image.open(self.data_dir / self.paths[i]) as im:
            img = fit_and_crop(im.convert("RGB"), self.crop_size, self.random_crop)
        if self.hflip and torch.rand(1).item() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.jpeg_quality is not None:
            img = recompress(img, self.jpeg_quality)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1)
        return x, torch.tensor(float(self.labels[i]))


def make_loader(manifest: pd.DataFrame, data_dir: str | Path, crop_size: int, batch_size: int, train: bool = False,
                num_workers: int = 0, jpeg_quality: int | None = None, augment: dict | None = None,
                seed: int = 0) -> DataLoader:
    """Shuffled, augmented loader for training; deterministic, order-preserving loader otherwise."""
    ds = ImageDataset(manifest, data_dir, crop_size, train=train, augment=augment, jpeg_quality=jpeg_quality)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
                      generator=torch.Generator().manual_seed(seed) if train else None,
                      drop_last=train and len(ds) > batch_size, pin_memory=torch.cuda.is_available(),
                      persistent_workers=num_workers > 0)


def to_float(x: torch.Tensor) -> torch.Tensor:
    """uint8 images (B, 3, H, W) -> float in [0, 1]."""
    return x.float().div(255.0)
