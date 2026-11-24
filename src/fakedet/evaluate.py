"""Evaluation: metrics with bootstrap 95% CIs, cross-generator evaluation of every trained run.

Metrics per (run, test set): accuracy, real accuracy, fake accuracy, ROC-AUC, average precision (AP).
Confidence intervals are percentile intervals of a stratified bootstrap over test images (reals and
fakes are resampled separately). Per-image logits are stored, so every later analysis (hypothesis
tests, tables) can recompute statistics without running a model again.

Outputs (results/):
    preds/<run>/<variant>__<generator>.npz  per-image logits + labels
    per_run/<run>.csv                       one row per test set for that run (resumable cache)
    cross_generator.csv                     all runs x test sets
    summary.csv                             mean accuracy over generators (all / unseen / separate) per run group
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from fakedet.config import (Config, generators, model_seeds, parse_run_name, planned_runs, separate_generators,
                            unseen_generators)
from fakedet.data.build import load_manifest, subset
from fakedet.inference import predict_logits
from fakedet.train import load_trained, runs_dir
from fakedet.utils import ensure_dir, get_device, log, save_json

METRICS = ["acc", "real_acc", "fake_acc", "auc", "ap"]


# ----------------------------------------------------------------- metrics
def batch_metrics(y: np.ndarray, S: np.ndarray, logit_thr: float = 0.0) -> dict:
    """Metrics for B score vectors at once. y: (N,) 0/1 labels shared by all rows, S: (B, N) logits."""
    y = np.asarray(y).astype(np.float64)
    S = np.atleast_2d(S).astype(np.float64)
    B, N = S.shape
    n_pos, n_neg = y.sum(), (1 - y).sum()
    pred = S >= logit_thr
    out = {"acc": (pred == (y > 0)).mean(1),
           "real_acc": (~pred[:, y == 0]).mean(1) if n_neg else np.full(B, np.nan),
           "fake_acc": pred[:, y == 1].mean(1) if n_pos else np.full(B, np.nan)}
    if n_pos == 0 or n_neg == 0:
        out["auc"] = out["ap"] = np.full(B, np.nan)
        return out
    ranks = rankdata(S, axis=1)  # average ranks for ties
    out["auc"] = ((ranks * y).sum(1) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    # average precision with tied scores grouped (same definition as sklearn)
    order = np.argsort(-S, axis=1, kind="stable")
    Ss, ys = np.take_along_axis(S, order, 1), y[order]
    tp = np.cumsum(ys, 1)
    precision = tp / np.arange(1, N + 1)
    is_last = np.ones_like(Ss, dtype=bool)
    is_last[:, :-1] = Ss[:, 1:] != Ss[:, :-1]
    idx = np.where(is_last, np.arange(N), N)
    group_end = np.minimum.accumulate(idx[:, ::-1], axis=1)[:, ::-1]
    out["ap"] = (ys * np.take_along_axis(precision, group_end, 1)).sum(1) / n_pos
    return out


def point_metrics(y: np.ndarray, s: np.ndarray, logit_thr: float = 0.0) -> dict:
    return {k: float(v[0]) for k, v in batch_metrics(y, s[None], logit_thr).items()}


def logit_threshold(p: float) -> float:
    return math.log(p / (1 - p))


def stratified_resample(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(B, N) indices into [reals..., fakes...] (labels must be sorted real-first, see `sort_by_label`)."""
    n_real = int((y == 0).sum())
    real = rng.integers(0, n_real, (n_boot, n_real))
    fake = rng.integers(n_real, len(y), (n_boot, len(y) - n_real))
    return np.concatenate([real, fake], axis=1)


def sort_by_label(y: np.ndarray, s: np.ndarray):
    o = np.argsort(y, kind="stable")
    return y[o], s[o]


def bootstrap_metrics(y: np.ndarray, s: np.ndarray, cfg: Config, seed: int = 0, chunk: int = 100) -> dict:
    """{metric: array (B,)} bootstrap replicates."""
    y, s = sort_by_label(np.asarray(y), np.asarray(s))
    rng = np.random.default_rng(seed)
    thr = logit_threshold(cfg.eval.threshold)
    parts = []
    for start in range(0, cfg.eval.bootstrap_samples, chunk):
        idx = stratified_resample(y, min(chunk, cfg.eval.bootstrap_samples - start), rng)
        parts.append(batch_metrics(y, s[idx], thr))
    return {k: np.concatenate([p[k] for p in parts]) for k in METRICS}


def ci_bounds(reps: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    return float(np.nanquantile(reps, a)), float(np.nanquantile(reps, 1 - a))


def metrics_with_ci(y: np.ndarray, s: np.ndarray, cfg: Config, seed: int = 0) -> dict:
    row = point_metrics(y, s, logit_threshold(cfg.eval.threshold))
    reps = bootstrap_metrics(y, s, cfg, seed)
    for k in METRICS:
        row[f"{k}_lo"], row[f"{k}_hi"] = ci_bounds(reps[k], cfg.eval.ci_level)
    return row


def stable_seed(*parts) -> int:
    return int(sum(ord(c) * (i + 1) for i, c in enumerate("|".join(map(str, parts))))) % (2 ** 31)


# ----------------------------------------------------------------- per-run evaluation
def results_dir(cfg: Config) -> Path:
    return ensure_dir(cfg.project.paths["results_dir"])


def preds_path(cfg: Config, run: str, variant: str, gen: str) -> Path:
    return Path(cfg.project.paths["results_dir"]) / "preds" / run / f"{variant}__{gen}.npz"


def load_preds(cfg: Config, run: str, variant: str, gen: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(preds_path(cfg, run, variant, gen))
    return d["label"], d["logit"]


def evaluate_run(cfg: Config, run: str, device: torch.device | None = None, force: bool = False) -> pd.DataFrame:
    """Evaluate one trained run on every (test variant x generator) test set."""
    cache = ensure_dir(results_dir(cfg) / "per_run") / f"{run}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    device = device or get_device(cfg.project.device)
    info = parse_run_name(run)
    model = load_trained(cfg, run, device)
    rows = []
    for variant in cfg.data.test_variants:
        manifest = load_manifest(cfg, variant)
        for gen in generators(cfg):
            df = subset(manifest, gen, "test")
            if df.empty:
                continue
            logits = predict_logits(model, df, cfg, device, desc=f"{run} -> {variant}/{gen}")
            y = df.label.to_numpy()
            p = preds_path(cfg, run, variant, gen)
            ensure_dir(p.parent)
            np.savez_compressed(p, label=y.astype(np.int8), logit=logits.astype(np.float32))
            rows.append({**info, "run": run, "test_variant": variant, "test_generator": gen, "n": len(df),
                         **metrics_with_ci(y, logits, cfg, stable_seed(cfg.project.seed, variant, gen))})
            log.info("%s -> %s/%s  acc %.3f [%.3f, %.3f]  auc %.3f", run, variant, gen, rows[-1]["acc"],
                     rows[-1]["acc_lo"], rows[-1]["acc_hi"], rows[-1]["auc"])
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def trained_runs(cfg: Config, models: list[str] | None = None) -> list[str]:
    return [r["name"] for r in planned_runs(cfg)
            if (not models or r["model"] in models) and (runs_dir(cfg) / r["name"] / "best.pt").exists()]


def evaluate_all(cfg: Config, models: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    runs = trained_runs(cfg, models)
    if not runs:
        raise RuntimeError("no trained runs found - run `fakedet train` first")
    device = get_device(cfg.project.device)
    frames = [evaluate_run(cfg, r, device, force) for r in runs]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(results_dir(cfg) / "cross_generator.csv", index=False)
    summarise(cfg, df)
    return df


# ----------------------------------------------------------------- summaries over generators / seeds
def mean_over_generators(cfg: Config, run: str, variant: str, gens: list[str]) -> dict | None:
    """Mean accuracy over `gens` for one run, with a bootstrap CI (generators resampled independently)."""
    reps, point = [], []
    for g in gens:
        if not preds_path(cfg, run, variant, g).exists():
            continue
        y, s = load_preds(cfg, run, variant, g)
        point.append(point_metrics(y, s, logit_threshold(cfg.eval.threshold))["acc"])
        reps.append(bootstrap_metrics(y, s, cfg, stable_seed(cfg.project.seed, variant, g))["acc"])
    if not reps:
        return None
    mean_rep = np.mean(reps, axis=0)
    lo, hi = ci_bounds(mean_rep, cfg.eval.ci_level)
    return {"acc": float(np.mean(point)), "lo": lo, "hi": hi, "n_generators": len(reps)}


def summarise(cfg: Config, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """results/summary.csv: per (model, train variant/generator, test variant), averaged over seeds."""
    if df is None:
        df = pd.read_csv(results_dir(cfg) / "cross_generator.csv")
    separate = separate_generators(cfg)
    all_gens = [g for g in generators(cfg) if g not in separate]
    unseen = [g for g in unseen_generators(cfg) if g not in separate]
    groups = {"mean_acc_all": all_gens, "mean_acc_unseen": unseen, **{f"acc_{g}": [g] for g in separate}}
    rows = []
    for (model, tv, tg, test_v), grp in df.groupby(["model", "train_variant", "train_generator", "test_variant"]):
        per_seed = []
        for seed, run in sorted(grp.groupby("seed").run.first().items()):
            r = {"seed": seed}
            for name, gens in groups.items():
                m = mean_over_generators(cfg, run, test_v, gens)
                if m:
                    r.update({name: m["acc"], f"{name}_lo": m["lo"], f"{name}_hi": m["hi"]})
            per_seed.append(r)
        s = pd.DataFrame(per_seed)
        row = {"model": model, "train_variant": tv, "train_generator": tg, "test_variant": test_v,
               "n_seeds": len(s)}
        for c in s.columns.drop("seed"):
            row[c] = s[c].mean()
        for name in groups:
            if name in s:
                row[f"{name}_seed_std"] = s[name].std(ddof=1) if len(s) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(results_dir(cfg) / "summary.csv", index=False)
    return out


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Per-generator metrics averaged over seeds (CI bounds averaged as well, std across seeds added)."""
    keys = ["model", "train_variant", "train_generator", "test_variant", "test_generator"]
    cols = [c for c in df.columns if any(c == m or c.startswith(m + "_") for m in METRICS)]
    g = df.groupby(keys)
    out = g[cols].mean()
    out["acc_seed_std"] = g["acc"].std(ddof=1).fillna(0.0)
    out["n_seeds"] = g.size()
    return out.reset_index()
