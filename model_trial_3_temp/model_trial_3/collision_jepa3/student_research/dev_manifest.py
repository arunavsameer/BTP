"""Deterministic dev subset: matched pairs per scenario group."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PATTERN_IDS = ("F1", "F2", "F3", "F4", "F5")


def _stable_key(*parts: str) -> int:
    h = hashlib.sha256(":".join(parts).encode()).hexdigest()
    return int(h[:12], 16)


def _row_tier(row: dict, gt_peak: float, stop_thr: float = 0.6, caution_thr: float = 0.35) -> str:
    if gt_peak >= stop_thr:
        return "STOP"
    if gt_peak >= caution_thr:
        return "THREAT"
    return "SAFE"


def _peak_bin(peak: float) -> int:
    return int(np.digitize(peak, [0.0, 0.05, 0.2, 0.5, 0.8, 1.01]))


def build_dev_manifest(
    rows: list[dict],
    gt_peaks: np.ndarray,
    *,
    dev_row_ids: list[int],
    dev_groups: set[str],
    pairs_per_group: int = 12,
    seed: int = 1337,
) -> dict[str, Any]:
    """Up to `pairs_per_group` complete pair anchors per dev scenario group."""
    by_group_pair: dict[str, dict[tuple, list[int]]] = defaultdict(lambda: defaultdict(list))
    for rid in dev_row_ids:
        row = rows[int(rid)]
        gid = str(row.get("scenario_group_id", ""))
        if gid not in dev_groups:
            continue
        pid = str(row.get("background_pair_id") or "")
        if not pid:
            continue
        t = int(row["frame"])
        by_group_pair[gid][(pid, t)].append(int(rid))

    selected: list[int] = []
    per_group_stats: dict[str, Any] = {}
    for gid in sorted(dev_groups):
        candidates = [(k, v) for k, v in by_group_pair.get(gid, {}).items() if len(v) == 2]
        # stratification bins
        scored: list[tuple[int, tuple, list[int]]] = []
        for (pid, t), idxs in candidates:
            r0 = rows[idxs[0]]
            peak = float(max(gt_peaks[i] for i in idxs))
            tier = _row_tier(r0, peak)
            pre = bool(r0.get("pre_fork"))
            bin_id = (_peak_bin(peak), tier, "pre" if pre else "post", t // 60)
            prio = _stable_key(str(seed), gid, pid, str(t), tier)
            scored.append((prio, bin_id, idxs))
        scored.sort(key=lambda x: x[0])
        # round-robin across bins
        bins: dict[tuple, list] = defaultdict(list)
        for prio, bin_id, idxs in scored:
            bins[bin_id].append((prio, idxs))
        picked_pairs = 0
        group_rows: list[int] = []
        while picked_pairs < pairs_per_group and bins:
            empty = []
            for bin_id in sorted(bins.keys(), key=str):
                if not bins[bin_id]:
                    empty.append(bin_id)
                    continue
                if picked_pairs >= pairs_per_group:
                    break
                _, idxs = bins[bin_id].pop(0)
                group_rows.extend(idxs)
                picked_pairs += 1
            for b in empty:
                del bins[b]
        group_rows = sorted(group_rows)
        selected.extend(group_rows)
        per_group_stats[gid] = {"n_pairs": picked_pairs, "n_rows": len(group_rows), "n_candidates": len(candidates)}

    selected = sorted(set(selected))
    n_pairs = len(selected) // 2
    return {
        "dev_row_ids": selected,
        "n_rows": len(selected),
        "n_pairs": n_pairs,
        "pairs_per_group_target": pairs_per_group,
        "n_groups": len(dev_groups),
        "per_group": per_group_stats,
        "seed": seed,
    }


def build_train_manifest(rows: list[dict], train_row_ids: list[int]) -> dict[str, Any]:
    by_pf: dict[tuple, list[int]] = defaultdict(list)
    for rid in train_row_ids:
        r = rows[int(rid)]
        pid = str(r.get("background_pair_id") or "")
        if pid:
            by_pf[(pid, int(r["frame"]))].append(int(rid))
    incomplete = [k for k, v in by_pf.items() if len(v) != 2]
    pairs = [k for k, v in by_pf.items() if len(v) == 2]
    return {
        "train_row_ids": sorted(int(i) for i in train_row_ids),
        "n_rows": len(train_row_ids),
        "n_pairs": len(pairs),
        "incomplete_pairs": len(incomplete),
        "pair_keys": [f"{a}:{t}" for a, t in sorted(pairs)],
    }


def manifest_integrity(
    rows: list[dict],
    train_manifest: dict,
    dev_manifest: dict,
    *,
    train_groups: set[str],
    dev_groups: set[str],
) -> dict[str, Any]:
    overlap = train_groups & dev_groups
    tr = set(train_manifest["train_row_ids"])
    dv = set(dev_manifest["dev_row_ids"])
    return {
        "ok": not overlap and not (tr & dv) and train_manifest["incomplete_pairs"] == 0,
        "group_overlap": sorted(overlap),
        "row_overlap_n": len(tr & dv),
        "n_train_rows": len(tr),
        "n_dev_rows": len(dv),
        "n_train_pairs": train_manifest["n_pairs"],
        "n_dev_pairs": dev_manifest["n_pairs"],
        "train_windows_is_800": len(tr) == 800,
    }
