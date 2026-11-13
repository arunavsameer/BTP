"""Development subset from training scenario groups (pilots only)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _order(groups: list[str], seed: int) -> list[str]:
    return sorted(groups, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())


def pilot_train_dev_groups(train_groups: list[str], seed: int = 1337, dev_fraction: float = 0.25) -> tuple[list[str], list[str]]:
    ordered = _order(train_groups, seed)
    n_dev = max(1, int(round(len(ordered) * dev_fraction)))
    dev = ordered[:n_dev]
    train = [g for g in ordered if g not in set(dev)]
    return train, dev


def write_split_manifest(path: Path, train: list[str], dev: list[str], seed: int, *, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"seed": seed, "pilot_train_groups": train, "pilot_dev_groups": dev}
    if extra:
        blob.update(extra)
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
