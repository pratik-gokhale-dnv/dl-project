"""Hypothesis tests H1-H4 from the proposal, computed from stored predictions (no model is run here).

H1  A frozen CLIP backbone (UnivFD) generalises best to unseen generators.
H2  Detectors trained on raw GenImage (JPEG reals vs PNG fakes) lose accuracy on format-matched data.
H3  The scratch CNN is less sensitive to JPEG re-compression than NPR.
H4  Accuracy falls as the spectral distance between training and test generator rises.

Intervals are percentile bootstrap intervals over test images (see evaluate.py); comparisons use the
same resamples for both sides when the test images are identical (paired bootstrap).
Verdicts: supported / partially supported / not supported / inconclusive (CI includes 0) / not evaluable.

Outputs: results/hypotheses.json and results/hypotheses.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fakedet.analysis.common import md_table, reference_run, spearman, transfer_pairs
from fakedet.analysis.fourier import load_spectral_distances
from fakedet.analysis.jpeg import jpeg_preds_path, load_jpeg_preds
from fakedet.config import Config, planned_runs, separate_generators, unseen_generators
from fakedet.evaluate import (acc_replicates, ci_bounds, logit_threshold, preds_path, run_acc_replicates,
                              stable_seed)
from fakedet.utils import log, save_json


def _runs(cfg: Config, model: str, variant: str, train_gens: list[str] | None = None) -> list[str]:
    return [r["name"] for r in planned_runs(cfg) if r["model"] == model and r["train_variant"] == variant
            and (train_gens is None or r["train_generator"] in train_gens)]


def _score(cfg: Config, runs: list[str], variant: str, test_gens: list[str]):
    """(mean accuracy, bootstrap replicates) averaged over runs x test generators; None if nothing is evaluated."""
    pts, reps = [], []
    for run in runs:
        for g in test_gens:
            if preds_path(cfg, run, variant, g).exists():
                p, r = run_acc_replicates(cfg, run, variant, g)
                pts.append(p)
                reps.append(r)
    return (float(np.mean(pts)), np.mean(reps, axis=0)) if reps else None


def _ci(cfg: Config, point: float, reps: np.ndarray) -> dict:
    lo, hi = ci_bounds(reps, cfg.eval.ci_level)
    return {"value": point, "lo": lo, "hi": hi}


def _title(cfg: Config, m: str) -> str:
    return cfg.models[m].get("title", m)


def _result(hid: str, claim: str, verdict: str, evidence: str, table: pd.DataFrame | None = None) -> dict:
    return {"id": hid, "claim": claim, "verdict": verdict, "evidence": evidence,
            "table": table if table is not None else pd.DataFrame()}


# ----------------------------------------------------------------- H1
def h1(cfg: Config) -> dict:
    claim = "Frozen CLIP (UnivFD) generalises best to unseen generators"
    clip_models = [m for m in cfg.analysis.hypotheses.h1.clip_models if m in cfg.experiments.models]
    others = [m for m in cfg.experiments.models if m not in clip_models]
    separate = separate_generators(cfg)
    unseen = [g for g in unseen_generators(cfg) if g not in separate]
    rows, scores = [], {}
    for m in cfg.experiments.models:
        runs = _runs(cfg, m, "matched")
        sc = _score(cfg, runs, "matched", unseen)
        if sc is None:
            continue
        scores[m] = sc
        row = {"model": _title(cfg, m), "group": "CLIP" if m in clip_models else "other", "runs": len(runs),
               "mean_acc_unseen": sc[0], **{k: v for k, v in zip(("ci_lo", "ci_hi"), ci_bounds(sc[1], cfg.eval.ci_level))}}
        for g in separate:
            s = _score(cfg, runs, "matched", [g])
            row[f"acc_{g}"] = s[0] if s else float("nan")
        rows.append((m, row))
    table = pd.DataFrame([r for _, r in rows]).sort_values("mean_acc_unseen", ascending=False)
    clip_have, other_have = [m for m in clip_models if m in scores], [m for m in others if m in scores]
    if not clip_have or not other_have:
        return _result("H1", claim, "not evaluable", "need at least one CLIP and one non-CLIP model with results", table)
    best_clip = max(clip_have, key=lambda m: scores[m][0])
    best_other = max(other_have, key=lambda m: scores[m][0])
    diff = scores[best_clip][1] - scores[best_other][1]
    d = _ci(cfg, scores[best_clip][0] - scores[best_other][0], diff)
    if d["value"] > 0 and d["lo"] > 0:
        verdict = "supported"
    elif d["value"] > 0:
        verdict = "inconclusive (best CLIP ahead, CI includes 0)"
    else:
        verdict = "not supported"
    ev = (f"Best CLIP variant {_title(cfg, best_clip)}: {scores[best_clip][0]:.3f} mean accuracy on unseen generators "
          f"({', '.join(unseen)}); best other detector {_title(cfg, best_other)}: {scores[best_other][0]:.3f}. "
          f"Difference {d['value']:+.3f} (95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}]).")
    return _result("H1", claim, verdict, ev, table)


# ----------------------------------------------------------------- H2
def h2(cfg: Config) -> dict:
    h = cfg.analysis.hypotheses.h2
    claim = "Raw-trained detectors lose accuracy on format-matched data (JPEG/PNG shortcut)"
    if not {"raw", "matched"} <= set(cfg.data.test_variants):
        return _result("H2", claim, "not evaluable", "data.test_variants must contain both raw and matched")
    g = h.train_generator
    rows, flags = [], []
    for m in cfg.experiments.models:
        for tv in ("raw", "matched"):
            runs = _runs(cfg, m, tv, [g])
            raw, mat = _score(cfg, runs, "raw", [g]), _score(cfg, runs, "matched", [g])
            if raw is None or mat is None:
                continue
            d = _ci(cfg, raw[0] - mat[0], raw[1] - mat[1])
            rows.append({"model": _title(cfg, m), "trained_on": tv, "acc_raw_test": raw[0], "acc_matched_test": mat[0],
                         "drop": d["value"], "drop_lo": d["lo"], "drop_hi": d["hi"]})
            if tv == "raw":
                flags.append((m, d["value"] >= h.min_drop and d["lo"] > 0))
    table = pd.DataFrame(rows)
    if not flags:
        return _result("H2", claim, "not evaluable", f"no raw-trained {g} runs with results", table)
    n_ok = sum(f for _, f in flags)
    verdict = "supported" if n_ok == len(flags) else "partially supported" if n_ok else "not supported"
    raw_rows = table[table.trained_on == "raw"]
    ev = (f"{n_ok}/{len(flags)} raw-trained detectors lose >= {h.min_drop:.2f} accuracy (CI lower bound > 0) when the same "
          f"generator ({g}) is tested on the format-matched version; mean drop {raw_rows['drop'].mean():+.3f}. "
          f"Matched-trained control models: mean drop {table[table.trained_on == 'matched']['drop'].mean():+.3f}.")
    return _result("H2", claim, verdict, ev, table)


# ----------------------------------------------------------------- H3
def _jpeg_drop(cfg: Config, run: str):
    """(point, replicates) of the accuracy drop averaged over test generators and JPEG qualities (paired)."""
    a = cfg.analysis.jpeg
    thr = logit_threshold(cfg.eval.threshold)
    pts, reps = [], []
    for g in a.generators:
        if not jpeg_preds_path(cfg, run, g, None).exists():
            continue
        seed = stable_seed("jpeg", g)
        y0, s0 = load_jpeg_preds(cfg, run, g, None)
        base_p = ((s0 >= thr) == (y0 > 0)).mean()
        base_r = acc_replicates(y0, s0, cfg, seed)
        for q in a.qualities:
            if jpeg_preds_path(cfg, run, g, q).exists():
                y, s = load_jpeg_preds(cfg, run, g, q)
                pts.append(base_p - ((s >= thr) == (y > 0)).mean())
                reps.append(base_r - acc_replicates(y, s, cfg, seed))
    return (float(np.mean(pts)), np.mean(reps, axis=0)) if reps else None


def h3(cfg: Config) -> dict:
    h = cfg.analysis.hypotheses.h3
    claim = "The scratch CNN is less sensitive to JPEG compression than NPR"
    rows, groups = [], {"cnn": [], "npr": []}
    for key, models in (("cnn", h.cnn_models), ("npr", h.npr_models)):
        for m in models:
            if m not in cfg.experiments.models:
                continue
            drop = _jpeg_drop(cfg, reference_run(cfg, m))
            if drop is None:
                continue
            groups[key].append(drop)
            rows.append({"model": _title(cfg, m), "group": key, "mean_acc_drop": drop[0],
                         **dict(zip(("ci_lo", "ci_hi"), ci_bounds(drop[1], cfg.eval.ci_level)))})
    table = pd.DataFrame(rows)
    if not groups["cnn"] or not groups["npr"]:
        return _result("H3", claim, "not evaluable", "run the jpeg analysis for the CNN and NPR models first", table)
    cnn = (np.mean([p for p, _ in groups["cnn"]]), np.mean([r for _, r in groups["cnn"]], axis=0))
    npr = (np.mean([p for p, _ in groups["npr"]]), np.mean([r for _, r in groups["npr"]], axis=0))
    d = _ci(cfg, npr[0] - cnn[0], npr[1] - cnn[1])
    verdict = ("supported" if d["lo"] > 0 else "not supported" if d["hi"] < 0
               else "inconclusive (CI includes 0)")
    ev = (f"Mean accuracy drop under JPEG re-compression (qualities {cfg.analysis.jpeg.qualities}, generators "
          f"{cfg.analysis.jpeg.generators}): CNN {cnn[0]:.3f}, NPR {npr[0]:.3f}. NPR minus CNN {d['value']:+.3f} "
          f"(95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}]); H3 predicts a positive difference.")
    return _result("H3", claim, verdict, ev, table)


# ----------------------------------------------------------------- H4
def h4(cfg: Config) -> dict:
    h = cfg.analysis.hypotheses.h4
    claim = "Accuracy falls as the spectral distance between training and test generator grows"
    try:
        dist = load_spectral_distances(cfg)
    except FileNotFoundError:
        return _result("H4", claim, "not evaluable", "run the fourier analysis first")
    train_gens = list(cfg.data.matched_train_generators)
    pairs = transfer_pairs(cfg, train_gens, h.get("include_separate", False))
    feat_corr = _load_feature_corr(cfg)
    rows = []
    for m in cfg.experiments.models:
        xs, ys = [], []
        for a, b in pairs:
            accs = [run_acc_replicates(cfg, r, "matched", b)[0] for r in _runs(cfg, m, "matched", [a])
                    if preds_path(cfg, r, "matched", b).exists()]
            if accs and (a, b) in dist:
                xs.append(dist[(a, b)])
                ys.append(np.mean(accs))
        rho, p = spearman(xs, ys)
        rows.append({"model": _title(cfg, m), "n_pairs": len(xs), "spearman_rho": rho, "p_value": p,
                     "feature_rho": feat_corr.get(m, {}).get("spearman_rho", float("nan")),
                     "feature_p": feat_corr.get(m, {}).get("p_value", float("nan"))})
    table = pd.DataFrame(rows)
    valid = table.dropna(subset=["spearman_rho"])
    if valid.empty:
        return _result("H4", claim, "not evaluable", "no (train, test) generator pairs with results", table)
    n_sig = int(((valid.spearman_rho < 0) & (valid.p_value < h.alpha)).sum())
    mean_rho = float(valid.spearman_rho.mean())
    if n_sig > len(valid) / 2:
        verdict = "supported"
    elif n_sig >= 1 and mean_rho < 0:
        verdict = "partially supported"
    else:
        verdict = "not supported"
    ev = (f"Spearman rho between spectral distance and accuracy over {int(valid.n_pairs.max())} (train, test) generator "
          f"pairs: mean {mean_rho:+.2f}; {n_sig}/{len(valid)} detectors show a significant negative correlation "
          f"(p < {h.alpha}). A negative rho means accuracy falls with distance.")
    return _result("H4", claim, verdict, ev, table)


def _load_feature_corr(cfg: Config) -> dict:
    p = Path(cfg.project.paths["results_dir"]) / "features" / "correlations.csv"
    return pd.read_csv(p).set_index("model").to_dict("index") if p.exists() else {}


# ----------------------------------------------------------------- driver
def run_hypotheses(cfg: Config) -> list[dict]:
    out = []
    for fn in (h1, h2, h3, h4):
        try:
            res = fn(cfg)
        except FileNotFoundError as e:
            res = _result(fn.__name__.upper(), "", "not evaluable", f"missing results: {e}")
        log.info("%s: %s", res["id"], res["verdict"])
        out.append(res)
    res_dir = Path(cfg.project.paths["results_dir"])
    save_json([{**r, "table": r["table"].to_dict("records")} for r in out], res_dir / "hypotheses.json")
    (res_dir / "hypotheses.md").write_text(hypotheses_markdown(out))
    return out


def hypotheses_markdown(results: list[dict]) -> str:
    parts = ["# Hypotheses\n"]
    for r in results:
        parts.append(f"## {r['id']}: {r['claim']}\n\n**Verdict: {r['verdict']}**\n\n{r['evidence']}\n")
        if len(r["table"]):
            parts.append(md_table(r["table"]) + "\n")
    return "\n".join(parts)
