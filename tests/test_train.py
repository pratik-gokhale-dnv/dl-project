from pathlib import Path

import pandas as pd
import pytest
import torch

from fakedet.config import run_name
from fakedet.train import load_trained, train_run, tune


@pytest.mark.parametrize("model", ["cnn", "clip_mid_mlp"])
def test_train_run_writes_artifacts_and_resumes(built_cfg, model):
    s = train_run(built_cfg, model, "matched", "sd14", 0, epochs=2)
    d = Path(built_cfg.project.paths["runs_dir"]) / run_name(model, "matched", "sd14", 0)
    for f in ("best.pt", "log.csv", "curves.png", "summary.json"):
        assert (d / f).exists(), f
    log = pd.read_csv(d / "log.csv")
    assert {"train_loss", "val_loss", "train_acc", "val_acc"} <= set(log.columns) and len(log) == s["epochs_run"]
    assert train_run(built_cfg, model, "matched", "sd14", 0) == s  # skipped, not retrained


def test_clip_checkpoint_is_head_only_and_reloads(built_cfg):
    train_run(built_cfg, "clip_final_linear", "matched", "sd14", 0, epochs=2)
    run = run_name("clip_final_linear", "matched", "sd14", 0)
    state = torch.load(f"{built_cfg.project.paths['runs_dir']}/{run}/best.pt")
    assert all(k.startswith("1.") or k.startswith("0.") for k in state)  # LayerNorm + Linear only
    assert load_trained(built_cfg, run, torch.device("cpu")).frozen_backbone


def test_tune_picks_lowest_val_loss(built_cfg):
    best = tune(built_cfg, models=["clip_final_linear"])
    grid = pd.read_csv(f"{built_cfg.project.paths['runs_dir']}/tuning/grid_results.csv")
    row = grid.loc[grid.val_loss.idxmin()]
    assert best["clip_final_linear"]["lr"] == row.lr


def test_scratch_models_learn_the_synthetic_signal(tmp_path):
    """Guards the whole data -> loader -> model -> trainer chain: a learnable task must be learned."""
    from fakedet.config import load_config
    from fakedet.data.build import build_datasets
    sets = [f"project.paths.{k}={tmp_path / k}" for k in ("data_dir", "runs_dir", "results_dir", "cache_dir")]
    cfg = load_config(Path(__file__).resolve().parents[1] / "config_smoke.yaml",
                      sets + ["data.synthetic.images_per_label=150", "models.cnn.training.epochs=12"])
    build_datasets(cfg)
    s = train_run(cfg, "cnn", "matched", "sd14", 0, use_tuned=False)
    assert s["best_val_acc"] >= 0.9 and s["best_val_loss"] < 0.5
