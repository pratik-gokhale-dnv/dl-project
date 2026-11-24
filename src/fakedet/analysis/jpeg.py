"""JPEG robustness (Hypothesis 3): accuracy of the reference detectors when test images are re-compressed.

For every detector trained on the reference run we re-encode test crops of the chosen generators at each
quality in `analysis.jpeg.qualities` and compare with the untouched images (`orig`). The same images
are used for every quality, so drops are paired.

Outputs (results/jpeg/): preds/<run>/<generator>__<quality|orig>.npz, jpeg_robustness.csv, jpeg_robustness.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fakedet.analysis.common import analysis_dir, head_per_label, reference_run, trained
from fakedet.config import Config
from fakedet.data.build import load_manifest, subset
from fakedet.evaluate import metrics_with_ci, stable_seed
from fakedet.inference import predict_logits
from fakedet.train import load_trained
from fakedet.utils import ensure_dir, get_device, log


def quality_tag(q) -> str:
    return "orig" if q is None else f"q{q}"


def jpeg_preds_path(cfg: Config, run: str, gen: str, q) -> Path:
    return Path(cfg.project.paths["results_dir"]) / "jpeg" / "preds" / run / f"{gen}__{quality_tag(q)}.npz"


def load_jpeg_preds(cfg: Config, run: str, gen: str, q) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(jpeg_preds_path(cfg, run, gen, q))
    return d["label"], d["logit"]


def run_jpeg(cfg: Config) -> pd.DataFrame:
    a = cfg.analysis.jpeg
    out = analysis_dir(cfg, "jpeg")
    device = get_device(cfg.project.device)
    manifest = load_manifest(cfg, a.variant)
    rows = []
    for name in cfg.experiments.models:
        run = reference_run(cfg, name)
        if not trained(cfg, run):
            log.warning("jpeg: %s not trained, skipping", run)
            continue
        model = None
        for g in a.generators:
            df = head_per_label(subset(manifest, g, "test"), a.n_images)
            for q in [None] + list(a.qualities):
                p = jpeg_preds_path(cfg, run, g, q)
                if p.exists():
                    y, logits = load_jpeg_preds(cfg, run, g, q)
                else:
                    model = model if model is not None else load_trained(cfg, run, device)
                    logits = predict_logits(model, df, cfg, device, jpeg_quality=q, desc=f"{run} {g} {quality_tag(q)}")
                    y = df.label.to_numpy()
                    ensure_dir(p.parent)
                    np.savez_compressed(p, label=y.astype(np.int8), logit=logits.astype(np.float32))
                rows.append({"model": name, "run": run, "generator": g, "quality": quality_tag(q),
                             **metrics_with_ci(y, logits, cfg, stable_seed("jpeg", g))})
        log.info("jpeg: %s done", run)
    df = pd.DataFrame(rows)
    base = df[df.quality == "orig"].set_index(["model", "generator"]).acc
    df["acc_drop"] = [base[(m, g)] - acc for m, g, acc in zip(df.model, df.generator, df.acc)]
    df.to_csv(out / "jpeg_robustness.csv", index=False)
    _plot(cfg, df, out)
    return df


def _plot(cfg: Config, df: pd.DataFrame, out) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = ["orig"] + [quality_tag(q) for q in cfg.analysis.jpeg.qualities]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for m, g in df.groupby("model"):
        mean = g.groupby("quality").acc.mean().reindex(order)
        ax.plot(order, mean.values, marker="o", label=cfg.models[m].get("title", m))
    ax.set(xlabel="test-time JPEG quality", ylabel=f"mean accuracy over {', '.join(cfg.analysis.jpeg.generators)}")
    ax.grid(alpha=.3), ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "jpeg_robustness.png", dpi=120)
    plt.close(fig)
