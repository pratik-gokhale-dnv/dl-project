"""Assemble results/report.md (tables + links to figures) and the accuracy heatmaps.

Everything here only reads files produced by the earlier stages, so it can be re-run at any time:
    fakedet report
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from fakedet.analysis.common import md_table, model_seed
from fakedet.config import Config, generators, run_name, separate_generators, title
from fakedet.evaluate import aggregate_seeds, summarise
from fakedet.utils import ensure_dir, log


def _ci(row, name: str) -> str:
    v = row.get(name)
    if v is None or pd.isna(v):
        return "-"
    lo, hi = row.get(f"{name}_lo"), row.get(f"{name}_hi")
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def main_table(cfg: Config, summ: pd.DataFrame, train_variant: str, test_variant: str) -> pd.DataFrame:
    sub = summ[(summ.train_variant == train_variant) & (summ.test_variant == test_variant)]
    cols = ["mean_acc_all", "mean_acc_unseen"] + [f"acc_{g}" for g in separate_generators(cfg)]
    rows = []
    for r in sub.to_dict("records"):
        row = {"model": cfg.models[r["model"]].get("title", r["model"]), "train gen": title(cfg, r["train_generator"]),
               "seeds": r["n_seeds"]}
        for c in cols:
            row[c.replace("acc_", "acc ")] = _ci(r, c)
        if r["n_seeds"] > 1 and not pd.isna(r.get("mean_acc_unseen_seed_std", np.nan)):
            row["unseen acc seed std"] = f"{r['mean_acc_unseen_seed_std']:.3f}"
        rows.append(row)
    order = {m: i for i, m in enumerate(cfg.experiments.models)}
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["_o"] = [order.get(next((k for k, v in cfg.models.items() if v.get("title", k) == t), t), 99) for t in out.model]
    return out.sort_values(["_o", "train gen"]).drop(columns="_o").fillna("-")


def per_generator_table(cfg: Config, df: pd.DataFrame, train_variant: str, train_gen: str, test_variant: str) -> pd.DataFrame:
    agg = aggregate_seeds(df)
    sub = agg[(agg.train_variant == train_variant) & (agg.train_generator == train_gen) & (agg.test_variant == test_variant)]
    rows = []
    for m in cfg.experiments.models:
        s = sub[sub.model == m].set_index("test_generator")
        if s.empty:
            continue
        row = {"model": cfg.models[m].get("title", m)}
        for g in generators(cfg):
            if g in s.index:
                r = s.loc[g]
                row[title(cfg, g)] = f"{r.acc:.3f} [{r.acc_lo:.3f}, {r.acc_hi:.3f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def heatmaps(cfg: Config, df: pd.DataFrame, fig_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    agg = aggregate_seeds(df)
    paths = []
    for test_variant in cfg.data.test_variants:
        sub = agg[agg.test_variant == test_variant]
        models = [m for m in cfg.experiments.models if m in set(sub.model)]
        if not models:
            continue
        cols = min(3, len(models))
        rows_n = (len(models) + cols - 1) // cols
        fig, ax = plt.subplots(rows_n, cols, figsize=(5.2 * cols, 2.8 * rows_n + 0.6), squeeze=False,
                               layout="constrained")
        gens = generators(cfg)
        for a, m in zip(ax.ravel(), models):
            s = sub[sub.model == m].copy()
            s["train"] = s.train_variant + "/" + s.train_generator
            piv = s.pivot(index="train", columns="test_generator", values="acc").reindex(columns=gens)
            im = a.imshow(piv.values, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
            a.set_xticks(range(len(gens)), [title(cfg, g) for g in gens], rotation=55, fontsize=6, ha="right")
            a.set_yticks(range(len(piv)), piv.index, fontsize=6)
            for (i, j), v in np.ndenumerate(piv.values):
                if not np.isnan(v):
                    a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)
            a.set_title(cfg.models[m].get("title", m), fontsize=8)
        for a in ax.ravel()[len(models):]:
            a.axis("off")
        fig.suptitle(f"Accuracy: training run (rows) x test generator (columns), {test_variant} test images", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.6)
        p = fig_dir / f"heatmap_{test_variant}.png"
        fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)
    return paths


def unseen_bars(cfg: Config, summ: pd.DataFrame, fig_dir: Path) -> Path | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = summ[(summ.train_variant == "matched") & (summ.test_variant == "matched")]
    if sub.empty or "mean_acc_unseen" not in sub:
        return None
    models = [m for m in cfg.experiments.models if m in set(sub.model)]
    gens = [g for g in cfg.data.matched_train_generators if g in set(sub.train_generator)]
    w = 0.8 / max(1, len(gens))
    fig, ax = plt.subplots(figsize=(1.0 * len(models) + 3, 3.6))
    for k, g in enumerate(gens):
        s = sub[sub.train_generator == g].set_index("model").reindex(models)
        err = np.vstack([s.mean_acc_unseen - s.mean_acc_unseen_lo, s.mean_acc_unseen_hi - s.mean_acc_unseen])
        ax.bar(np.arange(len(models)) + k * w, s.mean_acc_unseen, w, yerr=err, capsize=2, label=f"trained on {title(cfg, g)}")
    ax.set_xticks(np.arange(len(models)) + 0.4 - w / 2, [cfg.models[m].get("title", m) for m in models], rotation=40,
                  ha="right", fontsize=7)
    ax.set(ylabel="mean accuracy on unseen generators", ylim=(0.4, 1.0))
    ax.axhline(0.5, color="k", lw=.5, ls=":")
    ax.legend(fontsize=6), ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    p = fig_dir / "unseen_accuracy.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def run_report(cfg: Config) -> Path:
    res = Path(cfg.project.paths["results_dir"])
    df = pd.read_csv(res / "cross_generator.csv")
    summ = summarise(cfg, df)
    fig_dir = ensure_dir(res / "figures")
    figs = heatmaps(cfg, df, fig_dir)
    bar = unseen_bars(cfg, summ, fig_dir)
    rel = lambda p: os.path.relpath(p, res)  # noqa: E731
    ref = cfg.analysis.reference
    parts = ["# Results report\n", f"Generated from `{Path(cfg.config_path).name}`. Intervals are 95% bootstrap CIs over test images; "
             "multi-seed models show the mean over seeds.\n"]

    seen = Path(cfg.project.paths["data_dir"]) / "manifests" / "summary.csv"
    if seen.exists():
        parts += ["## Datasets\n", md_table(pd.read_csv(seen)), ""]

    for tv, tt in (("matched", "format-matched"), ("raw", "raw")):
        for test_v in cfg.data.test_variants:
            t = main_table(cfg, summ, tv, test_v)
            if len(t):
                parts += [f"## Trained on {tt} data, tested on {test_v} data\n",
                          "`mean acc all` excludes BigGAN; `mean acc unseen` = test-only generators excluding BigGAN.\n",
                          md_table(t), ""]
    if bar:
        parts += [f"![unseen accuracy]({rel(bar)})\n"]
    for p in figs:
        parts += [f"![{p.stem}]({rel(p)})\n"]

    for test_v in cfg.data.test_variants:
        t = per_generator_table(cfg, df, ref.train_variant, ref.train_generator, test_v)
        if len(t):
            parts += [f"## Per-generator accuracy: trained on {title(cfg, ref.train_generator)} ({ref.train_variant}), "
                      f"tested on {test_v}\n", md_table(t), ""]

    hyp = res / "hypotheses.md"
    if hyp.exists():
        parts += [hyp.read_text().replace("# Hypotheses", "## Hypotheses", 1), ""]

    parts.append("## Failure analysis\n")
    for name, desc in (("fourier/spectra.png", "Mean Fourier spectra of real/fake test images per generator"),
                       ("fourier/radial_fingerprints.png", "Radial spectral fingerprints and spectral distances"),
                       ("features/feature_distance_vs_acc.png", "Feature-space distance vs accuracy"),
                       ("jpeg/jpeg_robustness.png", "JPEG robustness")):
        if (res / name).exists():
            parts += [f"### {desc}\n", f"![{desc}]({name})\n"]
    cams = sorted((res / "gradcam").glob("*_examples.png")) if (res / "gradcam").exists() else []
    if cams:
        parts += ["### Grad-CAM\n"] + [f"![{c.stem}]({rel(c)})\n" for c in cams]
    curves = [Path(cfg.project.paths["runs_dir"]) / run_name(m, ref.train_variant, ref.train_generator, model_seed(cfg, m)) / "curves.png"
              for m in cfg.experiments.models]
    curves = [c for c in curves if c.exists()]
    if curves:
        parts += ["### Train / validation curves (reference runs)\n"] + [f"![{c.parent.name}]({rel(c)})\n" for c in curves]

    out = res / "report.md"
    out.write_text("\n".join(parts))
    log.info("wrote %s", out)
    return out
