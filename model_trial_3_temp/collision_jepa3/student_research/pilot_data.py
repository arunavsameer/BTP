"""Pilot row selection from train-pilot teacher cache."""

from __future__ import annotations

import json
from pathlib import Path

from collision_jepa3.data.splits import load_splits


def load_pilot_split_train(report_root: Path) -> dict:
    path = report_root / "pilot_split_train_groups.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; run scripts/49_student_pilot_train_setup.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def pilot_cache_split(cfg, cache_dir: Path, report_root: Path | None = None) -> tuple[str, list[str], list[str], int]:
    """Resolve pilot train/dev groups and cache-local row ids."""
    report_root = report_root or cache_dir.parent
    split = load_pilot_split_train(report_root)
    if split.get("source_partition") != "train":
        raise ValueError("pilot split must be built from train partition only")
    tr_g = list(split["pilot_train_groups"])
    dev_g = list(split["pilot_dev_groups"])
    overlap = set(tr_g) & set(dev_g)
    if overlap:
        raise ValueError(f"train/dev group overlap: {overlap}")
    if not split.get("cache_built"):
        raise RuntimeError("train pilot teacher cache not built; run scripts/50_student_pilot_train_cache.py")
    train_ids = list(split["train_row_ids"])
    dev_ids = list(split["dev_row_ids"])
    n = len(train_ids) + len(dev_ids)
    return "train_pilot", tr_g, dev_g, n


def pilot_row_ids(report_root: Path) -> tuple[list[int], list[int], list[str], list[str]]:
    split = load_pilot_split_train(report_root)
    return (
        list(split["train_row_ids"]),
        list(split["dev_row_ids"]),
        list(split["pilot_train_groups"]),
        list(split["pilot_dev_groups"]),
    )
