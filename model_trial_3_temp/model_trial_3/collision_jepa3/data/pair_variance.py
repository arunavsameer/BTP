"""Empirical variance of collision/miss heatmaps that share a pre-fork prefix."""

from __future__ import annotations

from typing import Any

import numpy as np

from .sampling import target_indices


def empirical_variance(maps: list[np.ndarray]) -> np.ndarray:
    """Population variance of equally weighted outcome maps. Two maps → (Hc−Hm)²/4."""
    if len(maps) < 2:
        raise ValueError("empirical_variance needs at least two maps")
    stack = np.stack([np.asarray(m, dtype=np.float64) for m in maps], axis=0)
    mu = stack.mean(axis=0, keepdims=True)
    return ((stack - mu) ** 2).mean(axis=0).astype(np.float32)


def envelope_max(maps: list[np.ndarray]) -> np.ndarray:
    stack = np.stack([np.asarray(m, dtype=np.float32) for m in maps], axis=0)
    return stack.max(axis=0)


def outcome_maps_at(
    heatmaps: dict[str, np.ndarray],
    mate_ids: list[str],
    frame_idx: int,
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for cid in mate_ids:
        hm = heatmaps.get(cid)
        if hm is None:
            continue
        if frame_idx < 0 or frame_idx >= hm.shape[0]:
            continue
        out.append(np.asarray(hm[frame_idx], dtype=np.float32))
    return out


def pair_targets_for_window(dataset, clip_id: str, anchor: int) -> dict[str, Any]:
    """Clip-own 1s map, pair variance, envelope. Variance is defined only pre-fork with ≥2 mates."""
    rec = dataset.records_by_id[clip_id]
    fork = int(rec.fork_frame or dataset.fork_frame)
    pre = int(anchor) < fork
    grid = int(dataset.cfg.get("data.target_grid", 5))
    zeros = np.zeros((grid, grid), dtype=np.float32)
    idx = int(target_indices(anchor, dataset.horizons)["1s"])
    hm = dataset._heatmaps.get(clip_id)
    if hm is None or idx < 0 or idx >= hm.shape[0]:
        gt_clip = zeros.copy()
    else:
        gt_clip = np.asarray(hm[idx], dtype=np.float32)
    gid = rec.outcome_group_id or ""
    mates = list(dataset._outcome_ids.get(gid, []))
    maps = outcome_maps_at(dataset._heatmaps, mates, idx)
    paired = bool(pre and len(maps) >= 2)
    if len(maps) >= 2:
        v = empirical_variance(maps)
        env = envelope_max(maps)
    else:
        v = zeros.copy()
        env = gt_clip.copy()
    if not paired:
        v = zeros.copy()
    collision_id = dataset._collision_ids.get(gid)
    canonical = bool(paired and collision_id and clip_id == collision_id)
    blob_n = int(np.sum(gt_clip >= 0.60))
    return {
        "gt_clip": gt_clip,
        "v_true": v,
        "envelope": env,
        "pre_fork": pre,
        "paired": paired,
        "canonical": canonical,
        "n_mates": int(len(maps)),
        "blob_n": blob_n,
        "target_idx": idx,
    }
