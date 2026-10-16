"""Group-level splits. Matching outcomes and backgrounds stay together."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .manifest import ClipRecord


def _hash_order(items: list[str], seed: int) -> list[str]:
    return sorted(items, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())


def _quota(n: int, train_fraction: float, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0
    n_val = int(round(n * val_fraction))
    n_test = int(round(n * test_fraction))
    n_val = max(1, n_val)
    n_test = max(1, n_test)
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        overflow = n_train + n_val + n_test - n
        while overflow > 0 and n_test > 1:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 1:
            n_val -= 1
            overflow -= 1
        n_train = n - n_val - n_test
    return n_train, n_val, n_test


def split_groups(
    records: list[ClipRecord],
    seed: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    stratify_by: str = "scenario_type",
) -> dict[str, list[str]]:
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    valid = [r for r in records if r.valid]
    group_stratum: dict[str, str] = {}
    strata: dict[str, list[str]] = defaultdict(list)
    for rec in valid:
        gid = rec.scenario_group_id
        if gid in group_stratum:
            continue
        key = str(getattr(rec, stratify_by, None) or rec.scenario_type)
        group_stratum[gid] = key
        strata[key].append(gid)

    assigned = {"train": [], "val": [], "test": []}
    leftovers: list[str] = []
    for _stratum, gids in sorted(strata.items()):
        ordered = _hash_order(sorted(set(gids)), seed)
        if len(ordered) >= 3:
            n_train, n_val, n_test = _quota(len(ordered), train_fraction, val_fraction, test_fraction)
            assigned["train"].extend(ordered[:n_train])
            assigned["val"].extend(ordered[n_train : n_train + n_val])
            assigned["test"].extend(ordered[n_train + n_val : n_train + n_val + n_test])
        else:
            leftovers.extend(ordered)

    leftovers = _hash_order(leftovers, seed + 17)
    n_train, n_val, n_test = _quota(len(leftovers), train_fraction, val_fraction, test_fraction)
    assigned["train"].extend(leftovers[:n_train])
    assigned["val"].extend(leftovers[n_train : n_train + n_val])
    assigned["test"].extend(leftovers[n_train + n_val : n_train + n_val + n_test])

    # If stratification left val/test empty, steal from the global pool.
    all_groups = _hash_order(sorted(group_stratum), seed + 31)
    if not assigned["val"] and len(all_groups) >= 2:
        donor = [g for g in all_groups if g in assigned["train"]]
        if donor:
            g = donor[-1]
            assigned["train"].remove(g)
            assigned["val"].append(g)
    if not assigned["test"] and len(all_groups) >= 3:
        donor = [g for g in all_groups if g in assigned["train"]]
        if donor:
            g = donor[-1]
            assigned["train"].remove(g)
            assigned["test"].append(g)

    for split in assigned:
        assigned[split] = sorted(set(assigned[split]))
    overlap = (
        set(assigned["train"]) & set(assigned["val"])
        | set(assigned["train"]) & set(assigned["test"])
        | set(assigned["val"]) & set(assigned["test"])
    )
    if overlap:
        raise RuntimeError(f"split leakage of groups {sorted(overlap)}")
    missing = set(group_stratum) - set(assigned["train"]) - set(assigned["val"]) - set(assigned["test"])
    if missing:
        raise RuntimeError(f"unassigned groups {sorted(missing)}")
    return assigned


def save_splits(splits: dict[str, list[str]], path: str | Path, extra: dict[str, Any] | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"splits": splits, **(extra or {})}
    text = json.dumps(payload, indent=2, sort_keys=True)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_splits(path: str | Path) -> dict[str, list[str]]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "splits" in data:
        return {k: list(v) for k, v in data["splits"].items()}
    return {k: list(v) for k, v in data.items() if isinstance(v, list)}


def clip_split_map(records: list[ClipRecord], splits: dict[str, list[str]]) -> dict[str, str]:
    group_to_split = {}
    for split, groups in splits.items():
        for g in groups:
            group_to_split[g] = split
    out = {}
    for rec in records:
        if rec.valid:
            out[rec.clip_id] = group_to_split[rec.scenario_group_id]
    return out
