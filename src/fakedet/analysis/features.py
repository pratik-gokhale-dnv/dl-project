"""Feature-space distance between generators vs. transfer accuracy.

For each detector trained on a (format-matched) training generator A we embed the test images of every
generator B with the detector's own pre-logit features, z-score them with the statistics of A's test
images (the detector's "home domain"), and measure

    d_feat(A, B) = || mean_fake_embedding(B) - mean_fake_embedding(A) || / sqrt(D)

We then ask whether accuracy on B drops as d_feat grows (Spearman correlation over (A, B) pairs).

Outputs (results/features/): distances.csv, correlations.csv, feature_distance_vs_acc.png
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fakedet.analysis.common import (analysis_dir, head_per_label, model_seed, spearman, transfer_pairs, trained)
from fakedet.config import Config, generators, run_name
from fakedet.data.build import load_manifest, subset
from fakedet.inference import extract_embeddings
from fakedet.train import load_trained
from fakedet.utils import get_device, log


def feature_distance(home: np.ndarray, other_fake: np.ndarray, home_fake: np.ndarray) -> float:
    mu, sd = home.mean(0), home.std(0) + 1e-6
    d = (other_fake.mean(0) - mu) / sd - (home_fake.mean(0) - mu) / sd
    return float(np.linalg.norm(d) / np.sqrt(home.shape[1]))


def run_features(cfg: Config) -> pd.DataFrame:
    a = cfg.analysis.features
    out = analysis_dir(cfg, "features")
    device = get_device(cfg.project.device)
    manifest = load_manifest(cfg, a.variant)
    acc_table = pd.read_csv(f"{cfg.project.paths['results_dir']}/cross_generator.csv")
    subsets = {g: head_per_label(subset(manifest, g, "test"), a.n_images) for g in generators(cfg)}
    rows = []
    for model_name in cfg.experiments.models:
        seed = model_seed(cfg, model_name)
        for train_gen in cfg.data.matched_train_generators:
            run = run_name(model_name, "matched", train_gen, seed)
            if not trained(cfg, run):
                continue
            model = load_trained(cfg, run, device)
            emb = {g: extract_embeddings(model, df, cfg, device, desc=f"{run}/{g}") for g, df in subsets.items()
                   if not df.empty}
            labels = {g: subsets[g].label.to_numpy() for g in emb}
            home, home_fake = emb[train_gen], emb[train_gen][labels[train_gen] == 1]
            acc = acc_table[(acc_table.run == run) & (acc_table.test_variant == a.variant)].set_index("test_generator").acc
            for g, e in emb.items():
                rows.append({"model": model_name, "train_generator": train_gen, "test_generator": g,
                             "feature_distance": feature_distance(home, e[labels[g] == 1], home_fake),
                             "acc": acc.get(g, np.nan)})
            log.info("features: %s done", run)
    df = pd.DataFrame(rows)
    df.to_csv(out / "distances.csv", index=False)
    corr = feature_correlations(cfg, df)
    corr.to_csv(out / "correlations.csv", index=False)
    _plot(cfg, df, out)
    return df


def feature_correlations(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    include = cfg.analysis.hypotheses.h4.get("include_separate", False)
    pairs = set(transfer_pairs(cfg, list(cfg.data.matched_train_generators), include))
    rows = []
    for model, g in df.groupby("model"):
        g = g[[(a, b) in pairs for a, b in zip(g.train_generator, g.test_generator)]]
        rho, p = spearman(g.feature_distance, g.acc)
        rows.append({"model": model, "n_pairs": len(g), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def _plot(cfg: Config, df: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = list(df.model.unique())
    if not models:
        return
    cols = min(3, len(models))
    fig, ax = plt.subplots((len(models) + cols - 1) // cols, cols, figsize=(3.6 * cols, 3 * ((len(models) + cols - 1) // cols)),
                           squeeze=False)
    for a_, m in zip(ax.ravel(), models):
        g = df[(df.model == m) & (df.train_generator != df.test_generator)]
        a_.scatter(g.feature_distance, g.acc, s=14, c=pd.factorize(g.train_generator)[0], cmap="tab10")
        a_.set(title=cfg.models[m].get("title", m), xlabel="feature distance", ylabel="accuracy", ylim=(0.4, 1.02))
        a_.grid(alpha=.3)
    for a_ in ax.ravel()[len(models):]:
        a_.axis("off")
    fig.tight_layout()
    fig.savefig(out / "feature_distance_vs_acc.png", dpi=120)
    plt.close(fig)
