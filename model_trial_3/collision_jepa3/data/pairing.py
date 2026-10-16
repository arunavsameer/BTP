"""Outcome-ambiguity masks and background-pair identity."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .manifest import ClipRecord
from .sampling import iter_anchors, target_indices


def heatmaps_disagree(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32)))) > float(tol)


def outcome_mates(records: Iterable[ClipRecord]) -> dict[str, list[ClipRecord]]:
    groups: dict[str, list[ClipRecord]] = defaultdict(list)
    for rec in records:
        if rec.valid and rec.outcome_group_id:
            groups[rec.outcome_group_id].append(rec)
    return groups


def background_mates(records: Iterable[ClipRecord]) -> dict[str, list[ClipRecord]]:
    groups: dict[str, list[ClipRecord]] = defaultdict(list)
    for rec in records:
        if rec.valid and rec.background_pair_id:
            groups[rec.background_pair_id].append(rec)
    return groups


def compute_ambiguity_table(
    clip_heatmaps: dict[str, np.ndarray],
    records: list[ClipRecord],
    *,
    fork_frame: int,
    n_frames: int,
    stride: int,
    offsets,
    horizons,
    horizon_names: list[str],
    heatmap_tol: float,
    mask_ambiguous: bool,
    allow_partial_horizons: bool = True,
) -> dict[tuple[str, int], dict[str, bool]]:
    """Return {(clip_id, anchor): {horizon_name: is_ambiguous}}."""
    table: dict[tuple[str, int], dict[str, bool]] = {}
    anchors = iter_anchors(
        n_frames,
        stride,
        offsets,
        horizons,
        allow_partial_horizons=allow_partial_horizons,
    )
    groups = outcome_mates(records)

    for rec in records:
        if not rec.valid:
            continue
        for t in anchors:
            table[(rec.clip_id, t)] = {name: False for name in horizon_names}

    if not mask_ambiguous:
        return table

    for _gid, mates in groups.items():
        if len(mates) < 2:
            continue
        # Pair collision against each miss sharing the same background.
        by_outcome = {m.outcome_type: m for m in mates}
        collision = by_outcome.get("collision")
        misses = [m for m in mates if m.outcome_type.startswith("miss")]
        if collision is None or not misses:
            continue
        h_col = clip_heatmaps.get(collision.clip_id)
        if h_col is None:
            continue
        fork = int(collision.fork_frame or fork_frame)
        for miss in misses:
            h_miss = clip_heatmaps.get(miss.clip_id)
            if h_miss is None:
                continue
            for t in anchors:
                history_identical = t < fork
                if not history_identical:
                    continue
                tg = target_indices(t, horizons)
                for name in horizon_names:
                    idx = tg[name]
                    if idx >= h_col.shape[0] or idx >= h_miss.shape[0]:
                        continue
                    if heatmaps_disagree(h_col[idx], h_miss[idx], heatmap_tol):
                        table[(collision.clip_id, t)][name] = True
                        table[(miss.clip_id, t)][name] = True
    return table


HORIZON_TARGET_KEY = {
    "1s": "h_1s",
    "1_5s": "h_1_5s",
    "2s": "h_2s",
}


def collision_clip_by_outcome_group(records: Iterable[ClipRecord]) -> dict[str, str]:
    """Map outcome_group_id → collision clip_id (same sample + background)."""
    out: dict[str, str] = {}
    for rec in records:
        if rec.valid and rec.outcome_type == "collision" and rec.outcome_group_id:
            out[rec.outcome_group_id] = rec.clip_id
    return out


def outcome_clip_ids_by_group(records: Iterable[ClipRecord]) -> dict[str, list[str]]:
    """Map outcome_group_id → clip_ids (collision + misses, same sample + background)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        if rec.valid and rec.outcome_group_id:
            groups[rec.outcome_group_id].append(rec.clip_id)
    return dict(groups)


def apply_collision_future_targets(
    targets: dict[str, np.ndarray],
    *,
    rec: ClipRecord,
    anchor: int,
    amb: dict[str, bool],
    clip_heatmaps: dict[str, np.ndarray],
    collision_ids: dict[str, str],
    horizons,
    horizon_names: list[str],
    available: dict[str, bool] | None = None,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Replace diverged future heatmaps with the paired collision clip's heatmap.

    Used when history is still pre-fork, so collision and miss look the same so
    far, but the future heatmap has already split. The collision map is the
    single training target. Returns the rewritten targets and the horizon names
    that were replaced.
    """
    gid = rec.outcome_group_id or ""
    col_id = collision_ids.get(gid)
    if not col_id or col_id not in clip_heatmaps:
        return targets, set()
    h_col = clip_heatmaps[col_id]
    tg_idx = target_indices(anchor, horizons)
    out = dict(targets)
    replaced: set[str] = set()
    for name in horizon_names:
        if not amb.get(name):
            continue
        if available is not None and not available.get(name, True):
            continue
        idx = tg_idx[name]
        if idx < 0 or idx >= h_col.shape[0]:
            continue
        key = HORIZON_TARGET_KEY.get(name, f"h_{name}")
        out[key] = np.asarray(h_col[idx], dtype=np.float32)
        replaced.add(name)
    return out, replaced


def apply_mean_future_targets(
    targets: dict[str, np.ndarray],
    *,
    rec: ClipRecord,
    anchor: int,
    amb: dict[str, bool],
    clip_heatmaps: dict[str, np.ndarray],
    outcome_ids: dict[str, list[str]],
    horizons,
    horizon_names: list[str],
    available: dict[str, bool] | None = None,
) -> tuple[dict[str, np.ndarray], set[str]]:
    """Replace diverged future heatmaps with the mean of collision and miss maps.

    Used when history is still pre-fork, so collision and miss look the same so
    far, but the future heatmap has already split. Both clips train on the
    cell-wise average. Returns the rewritten targets and the horizon names
    that were replaced.
    """
    gid = rec.outcome_group_id or ""
    mate_ids = outcome_ids.get(gid, [])
    if len(mate_ids) < 2:
        return targets, set()
    tg_idx = target_indices(anchor, horizons)
    out = dict(targets)
    replaced: set[str] = set()
    for name in horizon_names:
        if not amb.get(name):
            continue
        if available is not None and not available.get(name, True):
            continue
        idx = tg_idx[name]
        maps: list[np.ndarray] = []
        for cid in mate_ids:
            hm = clip_heatmaps.get(cid)
            if hm is None or idx < 0 or idx >= hm.shape[0]:
                continue
            maps.append(np.asarray(hm[idx], dtype=np.float32))
        if len(maps) < 2:
            continue
        key = HORIZON_TARGET_KEY.get(name, f"h_{name}")
        out[key] = np.mean(np.stack(maps, axis=0), axis=0).astype(np.float32)
        replaced.add(name)
    return out, replaced


def background_pair_valid(
    rec_a: ClipRecord,
    rec_b: ClipRecord,
    hm_a: np.ndarray,
    hm_b: np.ndarray,
    anchor: int,
    horizons,
    tol: float,
) -> bool:
    if rec_a.background_pair_id != rec_b.background_pair_id:
        return False
    if rec_a.clip_id == rec_b.clip_id:
        return False
    if rec_a.outcome_type != rec_b.outcome_type:
        return False
    if rec_a.scenario_group_id != rec_b.scenario_group_id:
        return False
    tg = target_indices(anchor, horizons)
    n = min(int(hm_a.shape[0]), int(hm_b.shape[0]))
    compared = 0
    for idx in tg.values():
        if idx < 0 or idx >= n:
            continue
        compared += 1
        if heatmaps_disagree(hm_a[idx], hm_b[idx], tol):
            return False
    return compared > 0
