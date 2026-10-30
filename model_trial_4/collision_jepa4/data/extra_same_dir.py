"""Optional same-dir extra clips (sample_0207..0229). Train-only; old val/test stay put."""

from __future__ import annotations

from typing import Iterable

from .manifest import ClipRecord


def sample_index(sample_id: str) -> int | None:
    parts = str(sample_id).split("_")
    if len(parts) >= 2 and parts[0] == "sample" and parts[1].isdigit():
        return int(parts[1])
    return None


def extra_same_dir_groups(records: Iterable[ClipRecord], cfg) -> set[str]:
    lo = int(cfg.get("data.extra_same_dir.sample_min", 207))
    hi = int(cfg.get("data.extra_same_dir.sample_max", 229))
    out: set[str] = set()
    for rec in records:
        if not rec.valid:
            continue
        idx = sample_index(rec.scenario_group_id)
        if idx is not None and lo <= idx <= hi:
            out.add(rec.scenario_group_id)
    return out


def extra_same_dir_records(records: Iterable[ClipRecord], cfg) -> list[ClipRecord]:
    groups = extra_same_dir_groups(records, cfg)
    return [r for r in records if r.valid and r.scenario_group_id in groups]


def apply_extra_same_dir_splits(
    splits: dict[str, list[str]],
    records: Iterable[ClipRecord],
    cfg,
) -> dict[str, list[str]]:
    """Strip extras from every split, then optionally add them to train.

    Default enabled=True. Does not rewrite splits.json. Val/test of the original
    pack stay comparable to gr40d.
    """
    extras = extra_same_dir_groups(records, cfg)
    out = {name: [g for g in groups if g not in extras] for name, groups in splits.items()}
    if not extras:
        return out
    if bool(cfg.get("data.extra_same_dir.enabled", True)):
        dest = str(cfg.get("data.extra_same_dir.split", "train"))
        out.setdefault(dest, [])
        out[dest] = sorted(set(out[dest]) | extras)
    return out
