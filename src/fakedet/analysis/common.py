"""Helpers shared by the analysis modules."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fakedet.config import Config, generators, model_seeds, run_name, separate_generators
from fakedet.data.dataset import fit_and_crop
from fakedet.utils import ensure_dir


def analysis_dir(cfg: Config, name: str) -> Path:
    return ensure_dir(Path(cfg.project.paths["results_dir"]) / name)


def head_per_label(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """First n images of each label (manifests are already randomly ordered, so this is a fixed random sample)."""
    return df.groupby("label", group_keys=False).head(n).reset_index(drop=True)


def load_crop(cfg: Config, rel_path: str, size: int | None = None) -> Image.Image:
    with Image.open(Path(cfg.project.paths["data_dir"]) / rel_path) as im:
        return fit_and_crop(im.convert("RGB"), size or cfg.data.crop_size, random_crop=False)


def reference_run(cfg: Config, model: str) -> str:
    r = cfg.analysis.reference
    return run_name(model, r.train_variant, r.train_generator, model_seed(cfg, model))


def trained(cfg: Config, run: str) -> bool:
    return (Path(cfg.project.paths["runs_dir"]) / run / "best.pt").exists()


def spearman(x, y) -> tuple[float, float]:
    """Spearman rho and p-value; (nan, nan) when it is undefined (< 3 points or constant input)."""
    from scipy.stats import spearmanr
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def transfer_pairs(cfg: Config, gens: list[str], include_separate: bool) -> list[tuple[str, str]]:
    """(train generator, test generator) pairs with train != test, optionally dropping BigGAN-like generators."""
    skip = set() if include_separate else set(separate_generators(cfg))
    return [(a, b) for a in gens for b in generators(cfg) if a != b and b not in skip]


def model_seed(cfg: Config, model: str) -> int:
    """Seed of the run used for single-run analyses: the reference seed when the model has it."""
    seeds = model_seeds(cfg, model)
    return cfg.analysis.reference.seed if cfg.analysis.reference.seed in seeds else seeds[0]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal GitHub-markdown table (avoids a `tabulate` dependency)."""
    def fmt(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
