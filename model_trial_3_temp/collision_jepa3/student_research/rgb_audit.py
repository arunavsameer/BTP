"""RGB / dense sampling audits (Part 2)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from collision_jepa3.data.cache import cache_signature_ok, clip_cache_dir
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.preprocessing import signature_from_cfg
from collision_jepa3.data.splits import load_splits
from collision_jepa3.data.sampling import FRAME_OFFSETS, HORIZONS_FRAMES, iter_anchors, valid_anchor_range


def audit_rgb(cfg) -> dict[str, Any]:
    records = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
    splits = load_splits(cfg.path("paths.split_file"))
    sig = signature_from_cfg(cfg)
    cache_root = cfg.path("paths.cache_root")
    report: dict[str, Any] = {"preprocess_signature": sig.as_dict(), "splits": {}, "issues": []}
    for split in ("train", "val"):
        groups = set(splits.get(split, []))
        recs = [r for r in records if r.scenario_group_id in groups]
        dims = Counter()
        fps_c = Counter()
        missing_video = []
        overlay_risk = []
        frame_cache_ok = 0
        frame_cache_miss = 0
        corrupt = []
        for r in recs:
            if "spatial_overlay" in (r.video_path or ""):
                overlay_risk.append(r.clip_id)
            vp = Path(r.video_path) if r.video_path else None
            if vp is None or not vp.exists():
                missing_video.append(r.clip_id)
            else:
                dims[(r.width or 1920, r.height or 1080)] += 1
                fps_c[float(r.fps or 30)] += 1
            cdir = clip_cache_dir(cache_root, r.clip_id)
            meta = cdir / "meta.json"
            npy = cdir / "frames_uint8.npy"
            if npy.exists() and meta.exists():
                meta_j = json.loads(meta.read_text())
                class _R:
                    video_path = r.video_path

                if cache_signature_ok(meta_j, sig, _R()) and npy.stat().st_size > 0:
                    frame_cache_ok += 1
                else:
                    corrupt.append(r.clip_id)
            else:
                frame_cache_miss += 1
        report["splits"][split] = {
            "n_clips": len(recs),
            "resolution_counts": {f"{w}x{h}": c for (w, h), c in dims.items()},
            "fps_counts": {str(k): v for k, v in fps_c.items()},
            "frame_cache_ok": frame_cache_ok,
            "frame_cache_miss": frame_cache_miss,
            "missing_video_n": len(missing_video),
            "overlay_path_n": len(overlay_risk),
            "corrupt_cache_n": len(corrupt),
        }
        if missing_video:
            report["issues"].append({"split": split, "missing_video_sample": missing_video[:20]})
    return report


def audit_dense(cfg, stride: int = 6) -> dict[str, Any]:
    n_frames = int(cfg.get("data.expected_frames", 450))
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
    anchors = list(iter_anchors(n_frames, stride, offsets, horizons, allow_partial_horizons=True))
    t_min, t_max = valid_anchor_range(n_frames, offsets, horizons, allow_partial_horizons=True)
    return {
        "stride_frames": stride,
        "anchors_per_clip": len(anchors),
        "valid_anchor_range": [t_min, t_max],
        "horizon_t_plus_30_required": True,
        "offsets_parent_16f": offsets,
        "note": "Student pilots use separate frame offsets (F1–F5); dense stride-6 anchors index teacher cache rows.",
        "storage_estimate_per_row_bytes": 2 * (5 * 5 + 32 * 5 * 5 + 128 * 16 * 16) + 64,
    }


def write_part2(out_dir: Path, cfg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb = audit_rgb(cfg)
    dense = audit_dense(cfg)
    (out_dir / "part2_rgb_availability.json").write_text(json.dumps(rgb, indent=2), encoding="utf-8")
    (out_dir / "part2_dense_sampling.json").write_text(json.dumps(dense, indent=2), encoding="utf-8")
