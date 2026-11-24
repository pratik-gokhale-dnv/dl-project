import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from fakedet.evaluate import (aggregate_seeds, batch_metrics, bootstrap_metrics, evaluate_all, point_metrics,
                              summarise)
from fakedet.train import train_run


@pytest.mark.parametrize("ties", [False, True])
def test_metrics_match_sklearn(ties):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    if ties:
        s = np.round(s)  # many identical scores
    m = point_metrics(y, s)
    assert m["auc"] == pytest.approx(roc_auc_score(y, s))
    assert m["ap"] == pytest.approx(average_precision_score(y, s))
    assert m["acc"] == pytest.approx(((s >= 0) == y).mean())
    assert m["real_acc"] == pytest.approx((s[y == 0] < 0).mean())
    assert m["fake_acc"] == pytest.approx((s[y == 1] >= 0).mean())


def test_batch_metrics_rows_are_independent():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(50), np.ones(50)]
    S = rng.normal(size=(4, 100)) + y
    batch = batch_metrics(y, S)
    for i in range(4):
        assert batch["auc"][i] == pytest.approx(roc_auc_score(y, S[i]))


def test_bootstrap_ci_contains_point_estimate(cfg):
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    s = rng.normal(size=400) + 1.5 * (2 * y - 1)
    reps = bootstrap_metrics(y, s, cfg)
    assert len(reps["acc"]) == cfg.eval.bootstrap_samples
    assert np.quantile(reps["acc"], 0.025) <= point_metrics(y, s)["acc"] <= np.quantile(reps["acc"], 0.975)


def test_evaluate_all_cross_generator_table(built_cfg):
    for model in ("cnn", "clip_final_linear"):
        train_run(built_cfg, model, "matched", "sd14", 0, epochs=2)
    df = evaluate_all(built_cfg, models=["cnn", "clip_final_linear"])
    n_gens = len(built_cfg.data.generators)
    assert len(df) == 2 * len(built_cfg.data.test_variants) * n_gens
    assert ((df.acc_lo <= df.acc) & (df.acc <= df.acc_hi)).all()
    s = summarise(built_cfg)
    assert {"mean_acc_all", "mean_acc_unseen", "acc_biggan", "mean_acc_all_lo"} <= set(s.columns)
    assert len(aggregate_seeds(df)) == len(df)
