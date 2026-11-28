"""Tiny synthetic generators so the whole pipeline can run offline (`data.source: synthetic`).

Reals are smooth random textures plus noise, stored as JPEG like GenImage's reals. Every fake generator adds its
own periodic fingerprint (a sinusoidal grating of a generator-specific frequency and a random phase) to the same
kind of texture and is stored as PNG, so the format shortcut, the format-matched rebuild and cross-generator
transfer all behave like the real data (at toy scale). Images are deterministic given `data.seed`.
"""
from __future__ import annotations

import io
import zlib

import numpy as np
import pandas as pd
from PIL import Image

from fakedet.config import Config
from fakedet.data import pool
from fakedet.utils import log

FINGERPRINT_AMPLITUDE = 14.0  # grey levels
OTHER_REAL_QUALITY = 85       # the oversampled extra reals are not Q96, like part of ImageNet


def fingerprint_frequency(cfg: Config, gen: str) -> tuple[float, float]:
    """Spatial frequency (cycles / pixel) of a generator's grating; distinct for each generator."""
    k = list(cfg.data.generators).index(gen)
    radius, angle = 0.10 + 0.045 * (k % 8), 0.9 * k
    return radius * np.cos(angle), radius * np.sin(angle)


def make_image(cfg: Config, gen: str, split: str, label: int, index: int) -> np.ndarray:
    size = cfg.data.synthetic.size
    rng = np.random.default_rng([cfg.data.seed, zlib.crc32(f"{gen}|{split}".encode()), label, index])
    coarse = Image.fromarray((rng.random((6, 6, 3)) * 255).astype(np.uint8))
    img = np.asarray(coarse.resize((size, size), Image.BICUBIC), dtype=np.float64) * 0.6 + 50
    img += rng.normal(0, 4, img.shape)
    if label:
        fx, fy = fingerprint_frequency(cfg, gen)
        yy, xx = np.mgrid[:size, :size]
        img += FINGERPRINT_AMPLITUDE * np.sin(2 * np.pi * (fx * xx + fy * yy) + rng.uniform(0, 2 * np.pi))[..., None]
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _encode(arr: np.ndarray, label: int, index: int, n_fake: int) -> bytes:
    buf = io.BytesIO()
    if label:
        Image.fromarray(arr).save(buf, "PNG")
    else:  # the first n_fake reals are Q96, the oversampled extras are not
        Image.fromarray(arr).save(buf, "JPEG", quality=96 if index < n_fake else OTHER_REAL_QUALITY)
    return buf.getvalue()


def populate_pool(cfg: Config, gen: str, split: str, n_real: int, n_fake: int, force: bool = False) -> pd.DataFrame:
    """Write n_real reals and n_fake fakes of `gen`/`split` into the pool (no-op if already complete)."""
    data_dir = cfg.project.paths["data_dir"]
    if force:
        pool.clear_pool(data_dir, gen, split)
    if pool.pool_exists(data_dir, gen, split):
        existing = pool.load_pool(data_dir, gen, split)
        if (existing.label == 0).sum() >= n_real and (existing.label == 1).sum() >= n_fake:
            return existing
        pool.clear_pool(data_dir, gen, split)
    rows = []
    for label, n in ((0, n_real), (1, n_fake)):
        for i in range(n):
            c = i % 10
            raw = _encode(make_image(cfg, gen, split, label, i), label, i, n_fake)
            rows.append(pool.add_image(data_dir, gen, split, len(rows), label, raw,
                                       f"n{c:08d}" if label == 0 else f"{c:03d}", src=f"synthetic/{gen}/{split}"))
    pool.save_pool(data_dir, gen, split, rows)
    log.info("%s/%s synthetic pool: %d real, %d fake", gen, split, n_real, n_fake)
    return pd.DataFrame(rows)
