"""Training: one generic loop for every detector, the run planner and hyper-parameter tuning.

* Image detectors (CNN / NPR / ResNet-50 / DeiT-S) train end-to-end on decoded crops.
* The frozen-CLIP detectors (UnivFD) train only their head, on cached backbone features.
Both share `fit`, so logging, schedule, early stopping and checkpointing are identical.

Every run writes to runs/<model>__<variant>__<generator>__s<seed>/:
    best.pt (trainable weights at the best validation loss), log.csv (per-epoch train/val loss + acc),
    curves.png, summary.json.   A run with an existing summary.json is skipped unless `force`.
"""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from fakedet.config import Config, model_training, planned_runs, run_name
from fakedet.data.build import load_manifest, subset
from fakedet.data.dataset import make_loader, to_float
from fakedet.inference import autocast_ctx, clip_features
from fakedet.models import build_model
from fakedet.utils import ensure_dir, get_device, load_json, log, save_json, seed_everything


def runs_dir(cfg: Config) -> Path:
    return Path(cfg.project.paths["runs_dir"])


def tuned_path(cfg: Config) -> Path:
    return runs_dir(cfg) / "tuning" / "best_hparams.json"


def load_tuned(cfg: Config) -> dict:
    p = tuned_path(cfg)
    return load_json(p) if p.exists() else {}


def make_optimizer(params, t: Config):
    kind = t.get("optimizer", "adamw").lower()
    if kind == "sgd":
        return torch.optim.SGD(params, lr=t.lr, momentum=0.9, weight_decay=t.weight_decay)
    cls = torch.optim.Adam if kind == "adam" else torch.optim.AdamW
    return cls(params, lr=t.lr, weight_decay=t.weight_decay)


def warmup_cosine(opt, total_steps: int, warmup_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def _plot_curves(log_df: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    for a, key, name in zip(ax, ("loss", "acc"), ("loss", "accuracy")):
        a.plot(log_df.epoch, log_df[f"train_{key}"], label="train")
        a.plot(log_df.epoch, log_df[f"val_{key}"], label="val")
        a.set_xlabel("epoch"), a.set_ylabel(name), a.grid(alpha=.3), a.legend()
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fit(model, forward, train_loader: DataLoader, val_loader: DataLoader, t: Config, device: torch.device,
        out_dir: Path, title: str = "") -> dict:
    """Train `model` (only parameters with requires_grad) and keep the best-validation-loss weights.

    forward(batch_x) -> logits (B,) is what is differentiated; it differs between raw-image and
    cached-feature training.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, t)
    steps = max(1, len(train_loader)) * t.epochs
    sched = warmup_cosine(opt, steps, int(t.get("warmup_epochs", 0) * len(train_loader)))
    use_amp = bool(t.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and not torch.cuda.is_bf16_supported())
    best, best_epoch, bad, rows = float("inf"), 0, 0, []
    ensure_dir(out_dir)

    for epoch in range(1, t.epochs + 1):
        t0 = time.time()
        model.train()
        tot = {"loss": 0.0, "correct": 0.0, "n": 0}
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx(device, use_amp):
                logits = forward(x)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, t.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = len(y)
            tot["loss"] += loss.item() * n
            tot["correct"] += ((logits.detach() > 0).float() == y).sum().item()
            tot["n"] += n
        val = evaluate_loop(model, forward, val_loader, device, use_amp)
        row = {"epoch": epoch, "train_loss": tot["loss"] / tot["n"], "train_acc": tot["correct"] / tot["n"],
               "val_loss": val["loss"], "val_acc": val["acc"], "lr": sched.get_last_lr()[0],
               "seconds": round(time.time() - t0, 1)}
        rows.append(row)
        log.info("%s epoch %d/%d  train %.4f/%.3f  val %.4f/%.3f", title, epoch, t.epochs, row["train_loss"],
                 row["train_acc"], row["val_loss"], row["val_acc"])
        if val["loss"] < best - 1e-6:
            best, best_epoch, bad = val["loss"], epoch, 0
            torch.save(model.trainable_state(), out_dir / "best.pt")
        else:
            bad += 1
            if t.get("patience", 0) and bad >= t.patience:
                log.info("%s early stop at epoch %d (best %d)", title, epoch, best_epoch)
                break

    log_df = pd.DataFrame(rows)
    log_df.to_csv(out_dir / "log.csv", index=False)
    _plot_curves(log_df, out_dir / "curves.png", title)
    model.load_trainable_state(torch.load(out_dir / "best.pt", map_location=device))
    return {"best_epoch": best_epoch, "best_val_loss": best, "epochs_run": len(rows),
            "best_val_acc": float(log_df.val_acc[best_epoch - 1]),
            "final_train_loss": rows[-1]["train_loss"]}


@torch.no_grad()
def evaluate_loop(model, forward, loader, device, use_amp: bool = False) -> dict:
    model.eval()
    loss = correct = n = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device, use_amp):
            logits = forward(x)
        loss += F.binary_cross_entropy_with_logits(logits.float(), y, reduction="sum").item()
        correct += ((logits > 0).float() == y).sum().item()
        n += len(y)
    return {"loss": loss / n, "acc": correct / n}


def _loaders(cfg: Config, model, variant: str, gen: str, t: Config, device: torch.device, seed: int):
    """(train_loader, val_loader, forward) for image models or cached-feature (CLIP) models."""
    manifest = load_manifest(cfg, variant)
    train_df, val_df = subset(manifest, gen, "train"), subset(manifest, gen, "val")
    if train_df.empty or val_df.empty:
        raise RuntimeError(f"no train/val images for generator {gen!r} in the {variant} manifest")
    if model.frozen_backbone:
        def to_loader(df, shuffle):
            feats = clip_features(model, df, cfg, device, desc=f"{gen}/{variant}")[model.layer].float()
            ds = TensorDataset(feats, torch.tensor(df.label.to_numpy(), dtype=torch.float32))
            g = torch.Generator().manual_seed(seed)
            return DataLoader(ds, batch_size=t.batch_size, shuffle=shuffle, generator=g,
                              drop_last=shuffle and len(ds) > t.batch_size)
        return to_loader(train_df, True), to_loader(val_df, False), lambda x: model.head(x).squeeze(1)
    common = dict(data_dir=cfg.project.paths["data_dir"], crop_size=cfg.data.crop_size,
                  batch_size=t.batch_size, num_workers=cfg.project.num_workers)
    train_loader = make_loader(train_df, train=True, augment=t.get("augment"), seed=seed, **common)
    val_loader = make_loader(val_df, train=False, **common)
    return train_loader, val_loader, lambda x: model(to_float(x))


def train_run(cfg: Config, model_name: str, variant: str, gen: str, seed: int, out_dir: Path | None = None,
              hparams: dict | None = None, epochs: int | None = None, force: bool = False,
              use_tuned: bool = True) -> dict:
    """Train one detector. `hparams` (e.g. {lr, weight_decay}) override the config for this run."""
    out_dir = Path(out_dir) if out_dir else runs_dir(cfg) / run_name(model_name, variant, gen, seed)
    summary_file = out_dir / "summary.json"
    if summary_file.exists() and not force:
        log.info("skip %s (already trained)", out_dir.name)
        return load_json(summary_file)

    t = model_training(cfg, model_name, load_tuned(cfg) if use_tuned else None)
    t.update(hparams or {})
    if epochs:
        t["epochs"] = epochs
    device = get_device(cfg.project.device)
    seed_everything(seed)
    model = build_model(cfg, model_name).to(device)
    train_loader, val_loader, forward = _loaders(cfg, model, variant, gen, t, device, seed)
    log.info("train %s: %d train / %d val images, %d trainable params, lr=%g wd=%g", out_dir.name,
             len(train_loader.dataset), len(val_loader.dataset), model.num_trainable(), t.lr, t.weight_decay)
    result = fit(model, forward, train_loader, val_loader, t, device, out_dir, title=out_dir.name)
    summary = {"model": model_name, "train_variant": variant, "train_generator": gen, "seed": seed,
               "n_train": len(train_loader.dataset), "n_val": len(val_loader.dataset),
               "trainable_params": model.num_trainable(),
               "hparams": {k: t[k] for k in ("lr", "weight_decay", "epochs", "batch_size", "optimizer")}, **result}
    save_json(summary, summary_file)
    return summary


def load_trained(cfg: Config, run: str, device: torch.device):
    """Rebuild a detector from its run directory (best validation checkpoint)."""
    from fakedet.config import parse_run_name
    info = parse_run_name(run)
    model = build_model(cfg, info["model"], load_pretrained=False)
    model.load_trainable_state(torch.load(runs_dir(cfg) / run / "best.pt", map_location="cpu"))
    return model.to(device).eval()


def train_all(cfg: Config, models: list[str] | None = None, generators: list[str] | None = None,
              variants: list[str] | None = None, force: bool = False) -> list[dict]:
    """Train every run in `planned_runs(cfg)` (optionally filtered)."""
    out = []
    todo = [r for r in planned_runs(cfg)
            if (not models or r["model"] in models) and (not generators or r["train_generator"] in generators)
            and (not variants or r["train_variant"] in variants)]
    for i, r in enumerate(todo, 1):
        log.info("=== run %d/%d: %s ===", i, len(todo), r["name"])
        out.append(train_run(cfg, r["model"], r["train_variant"], r["train_generator"], r["seed"], force=force))
    return out


# ----------------------------------------------------------------- hyper-parameter tuning
def tune(cfg: Config, models: list[str] | None = None, force: bool = False) -> dict:
    """Grid search (lr x weight_decay) per model on one generator; pick the best VALIDATION LOSS.
    Writes runs/tuning/best_hparams.json (read by later `train` calls when training.use_tuned)."""
    tcfg = cfg.tuning
    root = ensure_dir(runs_dir(cfg) / "tuning")
    best = load_json(tuned_path(cfg)) if tuned_path(cfg).exists() and not force else {}
    rows = []
    for model in models or cfg.experiments.models:
        keys = list(tcfg.grid.keys())
        results = []
        for combo in itertools.product(*(tcfg.grid[k] for k in keys)):
            hp = dict(zip(keys, combo))
            tag = "_".join(f"{k}{v:g}" for k, v in hp.items())
            s = train_run(cfg, model, tcfg.variant, tcfg.generator, 0, out_dir=root / model / tag, hparams=hp,
                          epochs=tcfg.epochs, force=force, use_tuned=False)
            results.append({"model": model, **hp, "val_loss": s["best_val_loss"], "val_acc": s["best_val_acc"]})
        rows += results
        pick = min(results, key=lambda r: r["val_loss"])
        best[model] = {k: pick[k] for k in keys}
        log.info("tuned %s -> %s (val loss %.4f)", model, best[model], pick["val_loss"])
    grid_file = root / "grid_results.csv"
    if grid_file.exists():  # keep rows of models that were not re-tuned now
        old = pd.read_csv(grid_file)
        rows = old[~old.model.isin({r["model"] for r in rows})].to_dict("records") + rows
    pd.DataFrame(rows).to_csv(grid_file, index=False)
    save_json(best, tuned_path(cfg))
    return best
