"""Group-mean H_1s targets (Pair-CF training-time rewrite)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from collision_jepa3.data.heatmap_annotations import load_heatmaps, targets_for_anchor
from collision_jepa3.data.manifest import ClipRecord, load_manifest
from collision_jepa3.data.pairing import apply_mean_future_targets
from collision_jepa3.data.sampling import HORIZONS_FRAMES


def _horizon_names() -> list[str]:
  return ["1s", "1_5s", "2s"]


def load_clip_heatmaps(cfg, clip_ids: set[str]) -> dict[str, np.ndarray]:
  records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
  n_frames = int(cfg.get("data.expected_frames", 450))
  out: dict[str, np.ndarray] = {}
  for cid in sorted(clip_ids):
    rec = records.get(cid)
    if rec is None:
      continue
    out[cid] = load_heatmaps(rec.heatmap_json_path, expected_frames=n_frames, grid=5)
  return out


def build_outcome_ids(records: list[ClipRecord]) -> dict[str, list[str]]:
  groups: dict[str, list[str]] = defaultdict(list)
  for r in records:
    if r.outcome_group_id:
      groups[r.outcome_group_id].append(r.clip_id)
  return dict(groups)


def build_group_mean_gt_array(
  cfg,
  cache_rows: list[dict],
  *,
  raw_gt: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
  """Return float32 [N,5,5] group-mean targets aligned to cache rows."""
  records = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
  rec_by_id = {r.clip_id: r for r in records}
  clip_ids = {str(r["clip_id"]) for r in cache_rows}
  heatmaps = load_clip_heatmaps(cfg, clip_ids)
  outcome_ids = build_outcome_ids(records)
  horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
  names = _horizon_names()
  n = len(cache_rows)
  gt = np.zeros((n, 5, 5), dtype=np.float32)
  amb_kept = 0
  replaced_n = 0
  missing = 0
  for i, row in enumerate(cache_rows):
    cid = str(row["clip_id"])
    t = int(row["frame"])
    if cid not in heatmaps or cid not in rec_by_id:
      missing += 1
      if raw_gt is not None:
        gt[i] = np.asarray(raw_gt[i], dtype=np.float32)
      continue
    tg = targets_for_anchor(heatmaps[cid], t, horizons, allow_missing=True)
    if tg.get("h_1s") is None:
      missing += 1
      if raw_gt is not None:
        gt[i] = np.asarray(raw_gt[i], dtype=np.float32)
      continue
    amb = {"1s": bool(row.get("amb_1s")), "1_5s": bool(row.get("amb_1_5s")), "2s": bool(row.get("amb_2s"))}
    tg2, replaced = apply_mean_future_targets(
      tg,
      rec=rec_by_id[cid],
      anchor=t,
      amb=amb,
      clip_heatmaps=heatmaps,
      outcome_ids=outcome_ids,
      horizons=horizons,
      horizon_names=names,
    )
    gt[i] = np.asarray(tg2["h_1s"], dtype=np.float32)
    if any(amb.values()):
      amb_kept += 1
    if replaced:
      replaced_n += 1
  meta = {
    "n_rows": n,
    "missing_or_fallback": missing,
    "amb_windows_retained": amb_kept,
    "group_mean_rewrites": replaced_n,
    "peak_max": float(gt.max()),
    "peak_mean": float(gt.mean()),
  }
  return gt, meta


def peak_tier(peak: float) -> str:
  if peak >= 0.60:
    return "STOP"
  if peak >= 0.25:
    return "THREAT"
  return "SAFE"
