"""Command line interface:  uv run fakedet <command> [--config config.yaml] [--set key=value ...]

Commands (run in this order by `fakedet all`, see `pipeline.stages` in the config):
    data      download/sample GenImage and build the raw + format-matched datasets
    tune      grid-search learning rate / weight decay per model (by validation loss)
    train     train every run planned in the config (model x training generator x seed)
    eval      evaluate every trained run on every test generator (metrics + 95% bootstrap CIs)
    analyze   failure analysis (Fourier, Grad-CAM, feature distances, JPEG) and hypotheses H1-H4
    report    assemble results/report.md
    all       the stages listed under pipeline.stages
    plan      list the planned runs and which are already trained
"""
from __future__ import annotations

import argparse
import sys

from fakedet.config import Config, load_config, planned_runs
from fakedet.utils import log

ANALYSES = ["fourier", "jpeg", "features", "gradcam", "hypotheses"]


def cmd_data(cfg: Config, args) -> None:
    from fakedet.data.build import build_datasets
    build_datasets(cfg, force=args.force)


def ensure_data(cfg: Config) -> None:
    """Stages that read images build the entire dataset first when it is missing."""
    from fakedet.data.build import ensure_data as build_if_missing
    build_if_missing(cfg)


def cmd_tune(cfg: Config, args) -> None:
    from fakedet.train import tune
    ensure_data(cfg)
    tune(cfg, models=args.models, force=args.force)


def cmd_train(cfg: Config, args) -> None:
    from fakedet.train import train_all
    ensure_data(cfg)
    train_all(cfg, models=args.models, generators=args.generators, variants=args.variants, force=args.force)


def cmd_eval(cfg: Config, args) -> None:
    from fakedet.evaluate import evaluate_all
    ensure_data(cfg)
    evaluate_all(cfg, models=args.models, force=args.force)


def cmd_analyze(cfg: Config, args) -> None:
    only = args.only or ANALYSES
    unknown = set(only) - set(ANALYSES)
    if unknown:
        raise SystemExit(f"unknown analysis {sorted(unknown)}; choose from {ANALYSES}")
    ensure_data(cfg)
    for name in [a for a in ANALYSES if a in only]:  # fixed order: later steps read earlier outputs
        log.info("=== analysis: %s ===", name)
        if name == "fourier":
            from fakedet.analysis.fourier import run_fourier
            run_fourier(cfg)
        elif name == "jpeg":
            from fakedet.analysis.jpeg import run_jpeg
            run_jpeg(cfg)
        elif name == "features":
            from fakedet.analysis.features import run_features
            run_features(cfg)
        elif name == "gradcam":
            from fakedet.analysis.gradcam import run_gradcam
            run_gradcam(cfg)
        else:
            from fakedet.analysis.hypotheses import run_hypotheses
            run_hypotheses(cfg)


def cmd_report(cfg: Config, args) -> None:
    from fakedet.analysis.report import run_report
    run_report(cfg)


def cmd_plan(cfg: Config, args) -> None:
    from pathlib import Path
    for r in planned_runs(cfg):
        done = (Path(cfg.project.paths["runs_dir"]) / r["name"] / "summary.json").exists()
        print(f"[{'x' if done else ' '}] {r['name']}")


def cmd_all(cfg: Config, args) -> None:
    for stage in cfg.pipeline.stages:
        if stage not in COMMANDS or stage == "all":
            raise SystemExit(f"pipeline.stages: unknown stage {stage!r}")
        log.info("######## stage: %s ########", stage)
        COMMANDS[stage](cfg, args)


COMMANDS = {"data": cmd_data, "tune": cmd_tune, "train": cmd_train, "eval": cmd_eval, "analyze": cmd_analyze,
            "report": cmd_report, "plan": cmd_plan, "all": cmd_all}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fakedet", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default="config.yaml", help="YAML config file (default: config.yaml)")
    common.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set training.epochs=3 (repeatable)")
    common.add_argument("--force", action="store_true", help="redo work even if its outputs already exist")
    common.add_argument("--models", nargs="+", help="restrict to these model names (tune/train/eval)")
    common.add_argument("--generators", nargs="+", help="restrict training to these training generators")
    common.add_argument("--variants", nargs="+", choices=["raw", "matched"], help="restrict training to these data variants")
    common.add_argument("--only", nargs="+", metavar="ANALYSIS", help=f"analyze: subset of {ANALYSES}")
    sub = p.add_subparsers(dest="command", required=True)
    for name, fn in COMMANDS.items():
        sub.add_parser(name, parents=[common], help=(fn.__doc__ or name).strip().split("\n")[0])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    COMMANDS[args.command](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
