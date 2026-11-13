"""Exact Pair-CF parent sampling contract for LatentMatch V3."""

from __future__ import annotations

import json
from pathlib import Path

from collision_jepa3.data.extra_same_dir import apply_extra_same_dir_splits
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.sampling import FRAME_OFFSETS, anchors_from_cfg
from collision_jepa3.data.splits import load_splits

OFFSETS_16 = list(FRAME_OFFSETS)
RAID = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")


def parent_split_groups(cfg) -> dict[str, set[str]]:
    recs = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
    splits = apply_extra_same_dir_splits(load_splits(cfg.path("paths.split_file")), recs, cfg)
    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    if train & val or train & test or val & test:
        raise RuntimeError("split group overlap")
    return {"train": train, "selection": val, "test": test}


def build_rows(cfg) -> list[dict]:
    recs = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
    groups = parent_split_groups(cfg)
    n_frames = int(cfg.get("data.expected_frames", 450))
    fork = int(cfg.get("sampling.fork_frame", 365))
    anchor_cache: dict[str, list[int]] = {}
    rows = []
    for split, gset in groups.items():
        recs_s = [r for r in recs if r.scenario_group_id in gset]
        recs_s.sort(key=lambda r: (r.clip_id, r.sample_id))
        for rec in recs_s:
            gid = rec.scenario_group_id
            if gid not in anchor_cache:
                anchor_cache[gid] = anchors_from_cfg(cfg, gid, n_frames=n_frames)
            for t in anchor_cache[gid]:
                rows.append(
                    {
                        "clip_id": rec.clip_id,
                        "sample_id": rec.sample_id,
                        "scenario_group_id": gid,
                        "frame": int(t),
                        "split": split,
                        "pre_fork": int(t) < fork,
                        "video_path": str(rec.video_path),
                        "background_id": rec.background_id,
                        "outcome": rec.outcome_type,
                        "outcome_group_id": rec.outcome_group_id,
                        "background_pair_id": rec.background_pair_id,
                    }
                )
    return rows


def summarize(rows: list[dict]) -> dict:
    from collections import defaultdict

    out = {}
    for split in ("train", "selection", "test"):
        rs = [r for r in rows if r["split"] == split]
        clips = {r["clip_id"] for r in rs}
        samples = {r["sample_id"] for r in rs}
        out[split] = {
            "windows": len(rs),
            "clips": len(clips),
            "sample_ids": len(samples),
            "pre_fork_windows": sum(1 for r in rs if r["pre_fork"]),
            "post_fork_windows": sum(1 for r in rs if not r["pre_fork"]),
        }
    return out


def write_sampling_contract(cfg, path: Path) -> dict:
    rows = build_rows(cfg)
    counts = summarize(rows)
    expected = {
        "train": {"windows": 22680, "clips": 567, "sample_ids": 159},
        "selection": {"windows": 5120, "clips": 128, "sample_ids": 32},
        "test": {"windows": 5120, "clips": 128, "sample_ids": 32},
    }
    for split, exp in expected.items():
        got = counts[split]
        for k, v in exp.items():
            if int(got[k]) != int(v):
                raise RuntimeError(f"{split} {k}={got[k]} expected {v}")
    blob = {
        "offsets": OFFSETS_16,
        "frames_per_window": 16,
        "timestamps_per_video": 40,
        "n_pre_fork": 26,
        "n_post_fork": 14,
        "anchor_mode": "group_random",
        "seed": 1337,
        "counts": counts,
        "expected": expected,
        "row_equality": "clip_id+frame+split from parent group_random anchors",
        "rows": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")
    return {"path": str(path), "counts": counts, "n_rows": len(rows)}
