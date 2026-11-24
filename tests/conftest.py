from pathlib import Path

import pytest

from fakedet.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg(tmp_path):
    """Smoke config (synthetic data, tiny models) with every output folder redirected to tmp_path."""
    sets = [f"project.paths.{k}={tmp_path / k}" for k in ("data_dir", "runs_dir", "results_dir", "cache_dir")]
    return load_config(ROOT / "config_smoke.yaml", sets)


@pytest.fixture(scope="session")
def built_cfg(tmp_path_factory):
    """Smoke config with the synthetic datasets already built (shared by the slower tests)."""
    from fakedet.data.build import build_datasets
    root = tmp_path_factory.mktemp("pipeline")
    sets = [f"project.paths.{k}={root / k}" for k in ("data_dir", "runs_dir", "results_dir", "cache_dir")]
    cfg = load_config(ROOT / "config_smoke.yaml", sets)
    build_datasets(cfg)
    return cfg
