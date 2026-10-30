"""Shared CLI helpers for model_trial_4 scripts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collision_jepa4.config import DEFAULT_CONFIG_PATH, Config


def add_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument("--dataset-root", default=None, help="Override paths.dataset_root")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def load_cfg(args) -> Config:
    overrides = {}
    if getattr(args, "dataset_root", None):
        overrides["paths.dataset_root"] = args.dataset_root
    if getattr(args, "seed", None) is not None:
        overrides["project.seed"] = args.seed
    return Config.load(args.config, overrides=overrides or None)


def resolve_checkpoint(cfg: Config, name: str) -> Path:
    root = cfg.path("paths.checkpoint_root")
    aliases = {
        "best": root / "teacher_phase_a_best.pt",
        "best_a": root / "teacher_phase_a_best.pt",
        "last": root / "teacher_phase_a_last.pt",
    }
    if name in aliases:
        return aliases[name]
    return Path(name)
