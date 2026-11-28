"""Build the datasets: fill the image pool, then derive the `raw` and `matched` versions with manifests.

    raw      images exactly as released (JPEG reals vs PNG fakes: file format is a shortcut).
    matched  format-matched after Grommelt et al.: only reals stored as JPEG quality `matched_quality` are kept and
             every fake is re-saved at that quality, so file format carries no information.
Both are class-balanced and have equal numbers of reals and fakes per (generator, split). Each training
generator's images are split into `train` / `val`; every generator also has a `test` split.

Outputs under <data_dir>/:  pool/ (original images), matched/ (re-saved fakes), manifests/{raw,matched,summary}.csv
A manifest row is one image: path (relative to data_dir), label, generator, split, cls, fmt, quality, width, height, bytes.
"""
from __future__ import annotations

import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from fakedet.config import Config, generators
from fakedet.data import genimage, pool, synthetic
from fakedet.data.formats import encode_jpeg
from fakedet.utils import ensure_dir, log

VARIANTS = ("raw", "matched")
MANIFEST_COLUMNS = ["path", "label", "generator", "split", "cls", "fmt", "quality", "width", "height", "bytes"]


# ----------------------------------------------------------------- manifests
def manifests_dir(cfg: Config) -> Path:
    return Path(cfg.project.paths["data_dir"]) / "manifests"


def manifest_path(cfg: Config, variant: str) -> Path:
    return manifests_dir(cfg) / f"{variant}.csv"


def load_manifest(cfg: Config, variant: str) -> pd.DataFrame:
    p = manifest_path(cfg, variant)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `fakedet data` to build the datasets")
    return pd.read_csv(p, dtype={"cls": str}, keep_default_na=False)


def subset(manifest: pd.DataFrame, gen: str, splits: str | list[str]) -> pd.DataFrame:
    """Rows of one generator in the given split(s), in manifest (random) order."""
    splits = [splits] if isinstance(splits, str) else list(splits)
    return manifest[(manifest.generator == gen) & manifest.split.isin(splits)].reset_index(drop=True)


# ----------------------------------------------------------------- what the config asks for
def train_generators(cfg: Config, variant: str) -> list[str]:
    return list(cfg.data.raw_train_generators if variant == "raw" else cfg.data.matched_train_generators)


def variant_splits(cfg: Config, variant: str) -> list[tuple[str, str]]:
    """(generator, split) pairs of a variant; splits are `train` (later divided into train/val) and `test`."""
    pairs = [(g, "train") for g in train_generators(cfg, variant)]
    if variant in cfg.data.test_variants:
        pairs += [(g, "test") for g in generators(cfg)]
    return pairs


def pool_splits(cfg: Config) -> list[tuple[str, str]]:
    """(generator, split) pairs whose original images must be in the pool."""
    training = set(cfg.data.raw_train_generators) | set(cfg.data.matched_train_generators)
    return [(g, s) for g in generators(cfg) for s in ("train", "test") if s == "test" or g in training]


def pool_targets(cfg: Config, split: str) -> tuple[int, int]:
    """(reals, fakes) to collect for a split; extra reals make up for those that are not JPEG-Q96."""
    d = cfg.data
    if d.source == "synthetic":
        n = d.synthetic.images_per_label if split == "train" else d.synthetic.test_images_per_label
    else:
        n = d.train_samples if split == "train" else d.test_samples
    return math.ceil(n * d.real_oversample), n


# ----------------------------------------------------------------- building
def build_pools(cfg: Config, force: bool = False) -> None:
    for gen, split in pool_splits(cfg):
        n_real, n_fake = pool_targets(cfg, split)
        if cfg.data.source == "synthetic":
            synthetic.populate_pool(cfg, gen, split, n_real, n_fake, force)
        else:
            genimage.collect_pool(cfg, gen, split, n_real, n_fake, force)


def _pick(df: pd.DataFrame, n: int, rng: random.Random) -> pd.DataFrame:
    return df.iloc[genimage.balanced_pick(df.cls.tolist(), n, rng)]


def _balanced_pairs(real: pd.DataFrame, fake: pd.DataFrame, rng: random.Random) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal numbers of reals and fakes, each class-balanced."""
    n = min(len(real), len(fake))
    return _pick(real, n, rng), _pick(fake, n, rng)


def _assign_splits(cfg: Config, df: pd.DataFrame, split: str, rng: random.Random) -> pd.DataFrame:
    """`train` images are divided into train/val per label (val_fraction); `test` stays test."""
    df = df.copy()
    df["split"] = split
    if split == "train":
        for label, idx in df.groupby("label").indices.items():
            idx = list(idx)
            rng.shuffle(idx)
            df.iloc[idx[:round(len(idx) * cfg.data.val_fraction)], df.columns.get_loc("split")] = "val"
    return df


def _resave_jpeg(src: Path, dst: Path, quality: int) -> dict:
    """Decode `src` and write it to `dst` as JPEG at `quality` (skipped if `dst` already exists)."""
    if not dst.exists():
        with Image.open(src) as im:
            raw = encode_jpeg(im, quality)
        ensure_dir(dst.parent)
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(raw)
        os.replace(tmp, dst)
    return pool.describe_image(dst.read_bytes())


def _matched_fakes(cfg: Config, fake: pd.DataFrame, gen: str, split: str) -> pd.DataFrame:
    data_dir, q = Path(cfg.project.paths["data_dir"]), cfg.data.matched_quality
    rel = [f"matched/{gen}/{split}/{Path(p).stem}.jpg" for p in fake.path]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
        infos = list(tqdm(ex.map(lambda a: _resave_jpeg(data_dir / a[0], data_dir / a[1], q), zip(fake.path, rel)),
                          total=len(rel), desc=f"re-save {gen}/{split} as JPEG-Q{q}", leave=False))
    out = fake.copy()
    out["path"] = rel
    for k in ("fmt", "quality", "width", "height"):
        out[k] = [i[k] for i in infos]
    out["bytes"] = [(data_dir / p).stat().st_size for p in rel]
    return out


def _variant_rows(cfg: Config, variant: str, gen: str, split: str) -> pd.DataFrame:
    p = pool.load_pool(cfg.project.paths["data_dir"], gen, split)
    real, fake = p[p.label == 0], p[p.label == 1]
    if variant == "matched":
        real = real[(real.fmt == "JPEG") & (real.quality == cfg.data.matched_quality)]
    rng = random.Random(f"{cfg.data.seed}|{variant}|{gen}|{split}")
    real, fake = _balanced_pairs(real, fake, rng)
    if variant == "matched":
        fake = _matched_fakes(cfg, fake, gen, split)
    wanted = pool_targets(cfg, split)[1]
    if len(real) < wanted:
        log.warning("%s %s/%s: only %d images per label (wanted %d)", variant, gen, split, len(real), wanted)
    df = pd.concat([real, fake]).drop(columns=["src"])
    df["generator"] = gen
    return _assign_splits(cfg, df, split, rng)


def _mode(s: pd.Series):
    return s.mode().iloc[0] if len(s) else float("nan")


def summarize(manifests: dict[str, pd.DataFrame], order: list[str]) -> pd.DataFrame:
    """Per (variant, generator, split) file-format statistics: makes the real/fake format shortcut visible."""
    rows = []
    for variant, m in manifests.items():
        for gen, split in [(g, s) for g in order for s in ("train", "val", "test")]:
            g = m[(m.generator == gen) & (m.split == split)]
            if g.empty:
                continue
            real, fake = g[g.label == 0], g[g.label == 1]
            rows.append({"variant": variant, "generator": gen, "split": split, "n_real": len(real), "n_fake": len(fake),
                         "real_jpeg_frac": (real.fmt == "JPEG").mean(), "fake_jpeg_frac": (fake.fmt == "JPEG").mean(),
                         "real_quality": _mode(real.quality), "fake_quality": _mode(fake.quality),
                         "real_kb": real.bytes.mean() / 1024, "fake_kb": fake.bytes.mean() / 1024,
                         "median_px": int(g.width.median())})
    return pd.DataFrame(rows).round(3)


def build_manifests(cfg: Config) -> dict[str, pd.DataFrame]:
    ensure_dir(manifests_dir(cfg))
    manifests = {}
    for variant in VARIANTS:
        frames = [_variant_rows(cfg, variant, g, s) for g, s in variant_splits(cfg, variant)]
        if not frames:
            continue
        m = pd.concat(frames, ignore_index=True)[MANIFEST_COLUMNS]
        manifests[variant] = m.sample(frac=1.0, random_state=cfg.data.seed).reset_index(drop=True)  # random order
        manifests[variant].to_csv(manifest_path(cfg, variant), index=False)
        log.info("%s manifest: %d images", variant, len(m))
    summarize(manifests, generators(cfg)).to_csv(manifests_dir(cfg) / "summary.csv", index=False)
    return manifests


def build_datasets(cfg: Config, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download / generate everything and write the manifests. Resumable and idempotent; `force` starts over."""
    log.info("building datasets in %s (source: %s)", cfg.project.paths["data_dir"], cfg.data.source)
    build_pools(cfg, force)
    return build_manifests(cfg)


# ----------------------------------------------------------------- on-demand build
def data_ready(cfg: Config) -> bool:
    """True when every manifest the config needs exists, covers every (generator, split) and all its files exist."""
    data_dir = Path(cfg.project.paths["data_dir"])
    for variant in VARIANTS:
        pairs = variant_splits(cfg, variant)
        if not pairs:
            continue
        if not manifest_path(cfg, variant).exists():
            return False
        m = load_manifest(cfg, variant)
        for gen, split in pairs:
            splits = ["train", "val"] if split == "train" else [split]
            if subset(m, gen, splits).empty:
                return False
        if not all((data_dir / p).exists() for p in m.path):
            return False
    return True


def ensure_data(cfg: Config) -> None:
    """Build the entire dataset (every generator, split and version in the config) unless it is already present."""
    if data_ready(cfg):
        return
    log.info("dataset not found or incomplete in %s - %s the entire dataset now", cfg.project.paths["data_dir"],
             "downloading" if cfg.data.source == "genimage" else "generating")
    build_datasets(cfg)
