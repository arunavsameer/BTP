#!/usr/bin/env python3
"""Write RGB + GT/pred H_now and H_1s comparison videos for test clips.

Uses preview-frame cache only (never spatial_overlay.mp4). Ground-truth heatmaps
are the clip's own annotations, not the mean_target training substitute.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg, resolve_checkpoint  # noqa: E402

from collision_jepa3.data.cache import load_cached_frames  # noqa: E402
from collision_jepa3.data.heatmap_annotations import load_heatmaps  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import input_indices, valid_anchor_range  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.visualizations import render_now_1s_strip  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

# Diverse test clips: collision vs miss on the same sample, plus harder vehicles.
PREFERRED_CLIP_IDS = [
    "sample_0003_jaywalker_turn_toward/collision/bg_00",
    "sample_0003_jaywalker_turn_toward/miss_left/bg_00",
    "sample_0016_child_darting/collision/bg_00",
    "sample_0098_car_cross_critical/collision/bg_00",
    "sample_0117_car_cut_in/collision/bg_00",
]


def _pick_clips(records, test_groups: set[str], n: int) -> list:
    by_id = {r.clip_id: r for r in records if r.valid and r.scenario_group_id in test_groups}
    picked = []
    for cid in PREFERRED_CLIP_IDS:
        rec = by_id.get(cid)
        if rec is not None:
            picked.append(rec)
        if len(picked) >= n:
            return picked
    # Fill from remaining test clips, preferring unused scenarios / outcomes.
    used_scen = {r.scenario_type for r in picked}
    used_out = {r.outcome_type for r in picked}
    rest = sorted(by_id.values(), key=lambda r: r.clip_id)
    for rec in rest:
        if rec.clip_id in {p.clip_id for p in picked}:
            continue
        if rec.scenario_type in used_scen and rec.outcome_type in used_out and len(picked) + (n - len(picked)) > 2:
            continue
        picked.append(rec)
        used_scen.add(rec.scenario_type)
        used_out.add(rec.outcome_type)
        if len(picked) >= n:
            break
    if len(picked) < n:
        for rec in rest:
            if rec not in picked:
                picked.append(rec)
            if len(picked) >= n:
                break
    return picked[:n]


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    """H.264 + faststart so Cursor/browser players open immediately (not OpenCV mp4v)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "fast", "-crf", "18", "-movflags", "+faststart",
            str(path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        for fr in frames:
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {path}")
        return
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


@torch.no_grad()
def render_clip(model, rec, cfg, device: str, *, stride: int, batch_size: int) -> list[np.ndarray]:
    cache_root = cfg.path("paths.cache_root")
    n_frames = int(cfg.get("data.expected_frames", 450))
    offsets = list(cfg.get("sampling.frame_offsets"))
    t_min, t_max = valid_anchor_range(
        n_frames,
        offsets,
        list(cfg.get("sampling.horizons_frames")),
        allow_partial_horizons=bool(cfg.get("sampling.allow_partial_horizons", True)),
    )
    frames_u8 = load_cached_frames(cache_root, rec.clip_id, mmap=True)
    heat = load_heatmaps(rec.heatmap_json_path, expected_frames=n_frames)
    fork = int(rec.fork_frame or cfg.get("sampling.fork_frame", 365))
    anchors = list(range(int(t_min), int(t_max) + 1, int(stride)))
    out_frames: list[np.ndarray] = []
    amp_dtype = torch.bfloat16 if device.startswith("cuda") else None

    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start : start + batch_size]
        clips = []
        for t in chunk:
            idx = input_indices(t, offsets)
            window = np.stack([np.asarray(frames_u8[i]) for i in idx], axis=0)
            clips.append(uint8_to_nchw01(window))
        x = torch.from_numpy(np.stack(clips, axis=0)).to(device)
        if amp_dtype is not None:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                pred = model(clip=x)
        else:
            pred = model(clip=x)
        h_now = pred["Hhat_now"].float().cpu().numpy()
        h_1s = pred["Hhat_1s"].float().cpu().numpy()
        for i, t in enumerate(chunk):
            rgb = np.asarray(frames_u8[t])
            tag = "pre-fork" if t < fork else "post-fork"
            title = (
                f"{rec.clip_id}  t={t}/{n_frames - 1}  {rec.outcome_type}  {tag}  "
                f"fork={fork}  best ckpt Phase A"
            )
            t1 = t + 30
            gt_1s = heat[t1] if t1 < n_frames else np.zeros_like(heat[t])
            panel = render_now_1s_strip(
                rgb,
                heat[t],
                h_now[i],
                gt_1s,
                h_1s[i],
                title=title,
            )
            out_frames.append(panel)
    return out_frames


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-clips", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1, help="Anchor stride inside each clip (1 = every frame)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = resolve_device(args.device)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    test_groups = set(splits.get(args.split, []))
    clips = _pick_clips(records, test_groups, args.n_clips)
    if len(clips) < args.n_clips:
        raise SystemExit(f"only found {len(clips)} {args.split} clips")

    model = CollisionTeacher3(cfg, dummy_backbone=bool(args.dummy))
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    blob = load_checkpoint(ckpt, map_location=device)
    if blob.get("lora_enabled") and not getattr(model.backbone, "lora_enabled", False):
        model.enable_lora()
    load_teacher_state(model, blob)
    model.to(device).eval()

    out_dir = cfg.path("paths.visualizations_root") / "test_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = float(cfg.get("data.fps_expected", 30)) / max(int(args.stride), 1)
    manifest = []
    for rec in clips:
        print(f"rendering {rec.clip_id} ...", flush=True)
        frames = render_clip(model, rec, cfg, device, stride=args.stride, batch_size=args.batch_size)
        safe = rec.clip_id.replace("/", "__")
        path = out_dir / f"{safe}_now_1s.mp4"
        _write_mp4(path, frames, fps)
        manifest.append(
            {
                "clip_id": rec.clip_id,
                "outcome_type": rec.outcome_type,
                "scenario_type": rec.scenario_type,
                "n_frames_written": len(frames),
                "path": str(path),
            }
        )
        print(f"  wrote {path} ({len(frames)} frames @ {fps:.1f} fps)", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"done: {len(manifest)} videos in {out_dir}")


if __name__ == "__main__":
    main()
