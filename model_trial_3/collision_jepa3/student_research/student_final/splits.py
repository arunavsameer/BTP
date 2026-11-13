"""Sample_id splits and data audit."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from collision_jepa3.data.splits import load_splits

from .targets import peak_tier


def load_cache_rows(cache_dir: Path) -> list[dict]:
  return [json.loads(l) for l in (cache_dir / "index.jsonl").open(encoding="utf-8")]


def dev_row_ids(report_dir: Path) -> list[int]:
  man = json.loads((report_dir / "part1_dev_manifest.json").read_text(encoding="utf-8"))
  return [int(x) for x in man["dev_row_ids"]]


def stratified_train_selection_split(
  sample_ids: list[str],
  stats: dict[str, dict[str, Any]],
  *,
  seed: int,
  selection_fraction: float = 0.2,
) -> tuple[list[str], list[str]]:
  """Deterministic stratified holdout by coarse label mix per sample."""
  buckets: dict[str, list[str]] = defaultdict(list)

  def bucket_key(sid: str) -> str:
    st = stats[sid]
    pf = "pre" if st.get("post_fork_windows", 0) == 0 else "mix" if st.get("pre_fork_windows", 0) else "post"
    return f"{st.get('dominant_tier','SAFE')}|{st.get('has_collision_bg',0)}|{pf}"

  for sid in sample_ids:
    buckets[bucket_key(sid)].append(sid)
  rng = random.Random(seed)
  train: list[str] = []
  selection: list[str] = []
  for _, members in sorted(buckets.items()):
    members = sorted(members)
    rng.shuffle(members)
    n_sel = max(0, int(round(selection_fraction * len(members))))
    if len(members) >= 2 and n_sel == 0:
      n_sel = 1
    selection.extend(members[:n_sel])
    train.extend(members[n_sel:])
  overlap = set(train) & set(selection)
  if overlap:
    raise ValueError(f"split overlap: {overlap}")
  return sorted(train), sorted(selection)


def build_data_splits(
  cfg,
  cache_dir: Path,
  report_dir: Path,
  gt_group_mean: np.ndarray,
  *,
  seed: int = 1337,
) -> dict[str, Any]:
  rows = load_cache_rows(cache_dir)
  dev_ids = set(dev_row_ids(report_dir))
  dev_sample_ids = {rows[i]["sample_id"] for i in dev_ids}
  official_train = set(load_splits(cfg.path("paths.split_file")).get("train", []))
  pool_ids = [
    i
    for i, r in enumerate(rows)
    if r["scenario_group_id"] in official_train and r["sample_id"] not in dev_sample_ids
  ]
  peaks = gt_group_mean.max(axis=(1, 2))
  per_sample: dict[str, dict[str, Any]] = defaultdict(
    lambda: {
      "windows": 0,
      "clips": set(),
      "pre_fork_windows": 0,
      "post_fork_windows": 0,
      "tier_counts": Counter(),
      "stop_windows": 0,
      "amb_1s_windows": 0,
      "backgrounds": set(),
      "outcomes": set(),
    }
  )
  for i in pool_ids:
    r = rows[i]
    sid = str(r["sample_id"])
    st = per_sample[sid]
    st["windows"] += 1
    st["clips"].add(r["clip_id"])
    if bool(r.get("pre_fork")):
      st["pre_fork_windows"] += 1
    else:
      st["post_fork_windows"] += 1
    tier = peak_tier(float(peaks[i]))
    st["tier_counts"][tier] += 1
    if tier == "STOP":
      st["stop_windows"] += 1
    if bool(r.get("amb_1s")):
      st["amb_1s_windows"] += 1
    st["backgrounds"].add(r.get("background_id", ""))
    st["outcomes"].add(r.get("outcome", ""))
  sample_stats: dict[str, dict[str, Any]] = {}
  for sid, st in per_sample.items():
    dominant = st["tier_counts"].most_common(1)[0][0] if st["tier_counts"] else "SAFE"
    sample_stats[sid] = {
      **{k: v for k, v in st.items() if k not in ("clips", "backgrounds", "outcomes", "tier_counts")},
      "n_clips": len(st["clips"]),
      "dominant_tier": dominant,
      "has_collision_bg": int("collision" in st["outcomes"]),
      "tier_counts": dict(st["tier_counts"]),
      "backgrounds": sorted(st["backgrounds"]),
      "outcomes": sorted(st["outcomes"]),
    }
  train_samples, sel_samples = stratified_train_selection_split(
    sorted(per_sample.keys()), sample_stats, seed=seed, selection_fraction=0.2
  )
  train_set, sel_set = set(train_samples), set(sel_samples)

  def assign(ids: list[int], part: str) -> list[int]:
    out = []
    for i in ids:
      sid = rows[i]["sample_id"]
      if part == "train" and sid in train_set:
        out.append(i)
      elif part == "selection" and sid in sel_set:
        out.append(i)
    return out

  train_row_ids = assign(pool_ids, "train")
  selection_row_ids = assign(pool_ids, "selection")
  assert not (set(train_row_ids) & set(selection_row_ids))
  assert not any(rows[i]["sample_id"] in dev_sample_ids for i in train_row_ids + selection_row_ids)

  def summarize(ids: list[int], name: str) -> dict[str, Any]:
    groups = set(rows[i]["sample_id"] for i in ids)
    stop_groups = {rows[i]["sample_id"] for i in ids if peaks[i] >= 0.60}
    pre = sum(1 for i in ids if rows[i].get("pre_fork"))
    post = len(ids) - pre
    tiers = Counter(peak_tier(float(peaks[i])) for i in ids)
    return {
      "name": name,
      "n_windows": len(ids),
      "n_sample_groups": len(groups),
      "n_stop_sample_groups": len(stop_groups),
      "pre_fork_windows": pre,
      "post_fork_windows": post,
      "tier_windows": dict(tiers),
      "amb_1s_windows": sum(1 for i in ids if rows[i].get("amb_1s")),
    }

  audit = {
    "seed": seed,
    "n_cache_rows": len(rows),
    "dev_windows": len(dev_ids),
    "dev_sample_groups": len(dev_sample_ids),
    "official_train_groups": len(official_train),
    "train_row_ids": train_row_ids,
    "selection_row_ids": selection_row_ids,
    "dev_row_ids": sorted(dev_ids),
    "train_samples": train_samples,
    "selection_samples": sel_samples,
    "train_summary": summarize(train_row_ids, "train"),
    "selection_summary": summarize(selection_row_ids, "selection"),
    "dev_summary": summarize(sorted(dev_ids), "dev"),
    "per_sample_stats": sample_stats,
    "split_integrity": {
      "train_selection_disjoint": True,
      "dev_disjoint_from_train_selection": True,
      "sample_id_unit": "sample_id",
    },
  }
  return audit
