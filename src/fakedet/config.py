"""Single-file configuration: load YAML, apply `--set a.b=c` overrides, validate."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access (cfg.data.crop_size)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = _wrap(v)

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(v):
    if isinstance(v, dict) and not isinstance(v, Config):
        return Config(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _unwrap(v):
    if isinstance(v, dict):
        return {k: _unwrap(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    return v


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(dict(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(raw: dict, overrides: list[str] | None) -> dict:
    """`--set training.epochs=2` -> raw['training']['epochs'] = 2 (value parsed as YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = raw
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                raise KeyError(f"Unknown config section {p!r} in override {item!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"Unknown config key {key!r} (typo?)")
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def load_config(path: str | Path = "config.yaml", overrides: list[str] | None = None) -> Config:
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw = apply_overrides(raw, overrides)
    cfg = Config(raw)
    cfg["config_path"] = str(path)
    for k, v in cfg.project.paths.items():
        p = Path(v)
        cfg.project.paths[k] = str(p if p.is_absolute() else path.parent / p)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    gens = cfg.data.generators
    if not gens:
        raise ValueError("data.generators is empty")
    for g, spec in gens.items():
        if spec.get("role") not in ("train", "test"):
            raise ValueError(f"data.generators.{g}.role must be 'train' or 'test'")
    for g in list(cfg.data.raw_train_generators) + list(cfg.data.matched_train_generators):
        if g not in gens or gens[g].role != "train":
            raise ValueError(f"{g!r} is listed as a training generator but has no role: train")
    for v in cfg.data.test_variants:
        if v not in ("raw", "matched"):
            raise ValueError(f"data.test_variants entries must be raw|matched, got {v!r}")
    for m in cfg.experiments.models:
        if m not in cfg.models:
            raise ValueError(f"experiments.models lists {m!r}, which is not defined under models:")
    for name, m in cfg.models.items():
        if m.get("type") not in ("cnn", "npr", "timm", "clip"):
            raise ValueError(f"models.{name}.type must be one of cnn|npr|timm|clip")
    if cfg.data.source not in ("genimage", "synthetic"):
        raise ValueError("data.source must be genimage|synthetic")


# ----------------------------------------------------------------- helpers
def generators(cfg: Config) -> list[str]:
    return list(cfg.data.generators.keys())


def unseen_generators(cfg: Config) -> list[str]:
    return [g for g, s in cfg.data.generators.items() if s.role == "test"]


def separate_generators(cfg: Config) -> list[str]:
    return [g for g, s in cfg.data.generators.items() if s.get("report_separately")]


def title(cfg: Config, gen: str) -> str:
    return cfg.data.generators[gen].get("title", gen)


def model_training(cfg: Config, model: str, tuned: dict | None = None) -> Config:
    """Global training defaults < per-model overrides < tuned hyper-parameters."""
    merged = deep_merge(cfg.training, cfg.models[model].get("training", {}))
    if tuned and cfg.training.get("use_tuned", True) and model in tuned:
        merged.update(tuned[model])
    return Config(merged)


def model_seeds(cfg: Config, model: str) -> list[int]:
    return list(cfg.models[model].get("seeds", cfg.experiments.seeds))


def run_name(model: str, variant: str, gen: str, seed: int) -> str:
    return f"{model}__{variant}__{gen}__s{seed}"


def parse_run_name(name: str) -> dict:
    model, variant, gen, seed = name.split("__")
    return {"model": model, "train_variant": variant, "train_generator": gen, "seed": int(seed[1:])}


def planned_runs(cfg: Config) -> list[dict]:
    """Every (model, variant, generator, seed) to train, from the config."""
    runs = []
    for model in cfg.experiments.models:
        for seed in model_seeds(cfg, model):
            for variant, gens in (("raw", cfg.data.raw_train_generators),
                                  ("matched", cfg.data.matched_train_generators)):
                for g in gens:
                    runs.append({"model": model, "train_variant": variant, "train_generator": g,
                                 "seed": seed, "name": run_name(model, variant, g, seed)})
    return runs
