"""Select dense train anchors with complete background pairs (~800 windows)."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def train_partition_groups(cfg) -> list[str]:
    from collision_jepa3.data.splits import load_splits

    return list(load_splits(cfg.path("paths.split_file")).get("train", []))


def build_train_dev_split(train_groups: list[str], seed: int = 1337, dev_fraction: float = 0.25) -> tuple[list[str], list[str]]:
    from collision_jepa3.student_research.splits_dev import pilot_train_dev_groups

    tr, dev = pilot_train_dev_groups(train_groups, seed=seed, dev_fraction=dev_fraction)
    overlap = set(tr) & set(dev)
    if overlap:
        raise ValueError(f"train/dev group overlap: {overlap}")
    return tr, dev


def select_pilot_row_ids(
    rows: list[dict],
    *,
    train_groups: set[str],
    dev_groups: set[str],
    target_train: int = 800,
    seed: int = 1337,
) -> dict[str, Any]:
    train_candidates = [i for i, r in enumerate(rows) if r.get("scenario_group_id") in train_groups]
    dev_ids = [i for i, r in enumerate(rows) if r.get("scenario_group_id") in dev_groups]
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for i in train_candidates:
        r = rows[i]
        pid = str(r.get("background_pair_id") or "")
        t = int(r["frame"])
        if pid:
            by_key[("pair", pid, t)].append(i)
        else:
            by_key[("solo", r["clip_id"], t)].append(i)
    pair_keys = [k for k, idxs in by_key.items() if k[0] == "pair" and len(idxs) == 2]
    solo_keys = [k for k in by_key if k[0] == "solo"]
    rng = random.Random(seed)
    rng.shuffle(pair_keys)
    selected: list[int] = []
    n_pairs = 0
    for k in pair_keys:
        selected.extend(by_key[k])
        n_pairs += 1
        if len(selected) >= target_train:
            break
    if len(selected) < target_train and solo_keys:
        rng.shuffle(solo_keys)
        for k in solo_keys:
            if len(selected) >= target_train:
                break
            selected.extend(by_key[k])
    selected = sorted(selected[:target_train])
    return {
        "train_row_ids": selected,
        "dev_row_ids": sorted(dev_ids),
        "n_train_target": target_train,
        "n_train_selected": len(selected),
        "n_dev": len(dev_ids),
        "n_complete_pairs_used": n_pairs,
        "skipped_incomplete_pairs": sum(1 for k, v in by_key.items() if k[0] == "pair" and len(v) != 2),
    }


def verify_manifest(rows: list[dict], manifest: dict[str, Any]) -> dict[str, Any]:
    tr = set(manifest["pilot_train_groups"])
    dv = set(manifest["pilot_dev_groups"])
    if tr & dv:
        return {"ok": False, "error": "group_overlap"}
    train_ids = manifest["train_row_ids"]
    dev_ids = manifest["dev_row_ids"]
    for name, ids, groups in (
        ("train", train_ids, tr),
        ("dev", dev_ids, dv),
    ):
        for rid in ids:
            r = rows[int(rid)]
            if r.get("split") != "train":
                return {"ok": False, "error": f"{name}_row_not_train_split", "row_id": rid}
            if r.get("scenario_group_id") not in groups:
                return {"ok": False, "error": f"{name}_row_wrong_group", "row_id": rid}
    # pair completeness on train
    by_pf: dict[tuple, set[str]] = defaultdict(set)
    for rid in train_ids:
        r = rows[int(rid)]
        pid = str(r.get("background_pair_id") or "")
        if not pid:
            continue
        by_pf[(pid, int(r["frame"]))].add(r["clip_id"])
    bad_pairs = [k for k, clips in by_pf.items() if len(clips) != 2]
    return {
        "ok": len(bad_pairs) == 0,
        "n_train_rows": len(train_ids),
        "n_dev_rows": len(dev_ids),
        "incomplete_pairs_in_train": bad_pairs[:10],
        "n_incomplete_pairs": len(bad_pairs),
    }


def write_pilot_split_train_groups(path: Path, *, seed: int, train_groups: list[str], dev_groups: list[str], row_manifest: dict) -> None:
    blob = {
        "seed": seed,
        "source_partition": "train",
        "pilot_train_groups": train_groups,
        "pilot_dev_groups": dev_groups,
        "note": "Development groups are from the original training partition only; official val is never used for pilots.",
        **row_manifest,
    }
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
