#!/usr/bin/env python3
"""Render RGB + pred/GT 1s-heatmap videos for direct_h1s checkpoints.

Uses preview.mp4 only (never spatial_overlay.mp4). Letterbox, no FOV crop.
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
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.heatmap_annotations import load_heatmaps  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import preprocess_clip, signature_from_cfg, uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import input_indices, valid_anchor_range  # noqa: E402
from collision_jepa3.data.video_reader import decode_all_rgb  # noqa: E402
from collision_jepa3.evaluation.visualizations import heatmap_to_image, overlay_heatmap  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

DEFAULT_CLIP = "sample_0003_jaywalker_turn_toward/collision/bg_00"
DEFAULT_CKPTS = [
    "gr40b=checkpoints/direct_h1s_appear_blob_gr40b/teacher_phase_a_best.pt",
    "gr40c=checkpoints/direct_h1s_appear_blob_gr40c/teacher_phase_a_best.pt",
    "gr40d=checkpoints/direct_h1s_appear_blob_gr40d/teacher_phase_a_best.pt",
]
STILL_TIMES = (180, 300, 360, 390, 410)
CAUTION, STOP_T = 0.25, 0.60
PANEL = 320
HEADER = 52


def _band(peak: float) -> str:
    if peak >= STOP_T:
        return "STOP"
    if peak >= CAUTION:
        return "THREAT"
    return "SAFE"


def _font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
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
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()


def _tile(rgb: np.ndarray, overlay: np.ndarray, pred: np.ndarray, gt: np.ndarray, title: str) -> np.ndarray:
    def up(im: np.ndarray) -> Image.Image:
        return Image.fromarray(im).resize((PANEL, PANEL), Image.BILINEAR)

    tiles = [
        ("RGB t", up(rgb)),
        ("pred overlay", up(overlay)),
        ("pred H_1s", heatmap_to_image(pred).resize((PANEL, PANEL), Image.NEAREST)),
        ("GT H_1s", heatmap_to_image(gt).resize((PANEL, PANEL), Image.NEAREST)),
    ]
    canvas = Image.new("RGB", (PANEL * 4, PANEL + HEADER), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    small = _font(14)
    draw.text((8, 6), title, fill=(240, 240, 240), font=font)
    for i, (lab, im) in enumerate(tiles):
        x = i * PANEL
        draw.text((x + 8, 28), lab, fill=(210, 210, 210), font=small)
        canvas.paste(im, (x, HEADER))
    return np.asarray(canvas, dtype=np.uint8)


@torch.no_grad()
def render_one(model, frames_u8, heat, rec, cfg, device: str, name: str, stride: int, batch_size: int):
    n_frames = int(frames_u8.shape[0])
    offsets = list(cfg.get("sampling.frame_offsets"))
    t_min, t_max = valid_anchor_range(
        n_frames,
        offsets,
        list(cfg.get("sampling.horizons_frames")),
        allow_partial_horizons=bool(cfg.get("sampling.allow_partial_horizons", True)),
    )
    fork = int(rec.fork_frame or cfg.get("sampling.fork_frame", 365))
    anchors = list(range(int(t_min), int(t_max) + 1, int(stride)))
    out_frames: list[np.ndarray] = []
    stills: dict[int, np.ndarray] = {}
    rows = []
    amp = torch.bfloat16 if str(device).startswith("cuda") else None
    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start : start + batch_size]
        clips = []
        for t in chunk:
            idx = input_indices(t, offsets)
            window = np.stack([np.asarray(frames_u8[i]) for i in idx], axis=0)
            clips.append(uint8_to_nchw01(window))
        x = torch.from_numpy(np.stack(clips, axis=0)).to(device)
        if amp is not None:
            with torch.autocast(device_type="cuda", dtype=amp):
                pred = model(clip=x)
        else:
            pred = model(clip=x)
        h_1s = pred["Hhat_1s"].float().cpu().numpy()
        for i, t in enumerate(chunk):
            rgb = np.asarray(frames_u8[t])
            hm = h_1s[i]
            peak = float(hm.max())
            label = _band(peak)
            t1 = t + 30
            gt = heat[t1] if t1 < n_frames else np.zeros((5, 5), dtype=np.float32)
            tag = "pre-fork" if t < fork else "post-fork"
            overlay = overlay_heatmap(rgb, hm, alpha=0.50)
            title = (
                f"{name}  {rec.clip_id}  t={t} ({t / 30:.1f}s)  {rec.outcome_type}  "
                f"{tag}  peak={peak:.2f}  {label}  (H_1s = danger in 1s)"
            )
            panel = _tile(rgb, overlay, hm, gt, title)
            out_frames.append(panel)
            if t in STILL_TIMES:
                stills[t] = panel
            rows.append({"t": t, "peak": peak, "label": label, "pre_fork": t < fork})
    return out_frames, stills, rows


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--clip-id", default=DEFAULT_CLIP)
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CKPTS)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out-dir", default="visualizations/gr40_compare")
    args = parser.parse_args()

    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    device = resolve_device(args.device)
    records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    rec = records.get(args.clip_id)
    if rec is None:
        raise SystemExit(f"clip not in manifest: {args.clip_id}")
    if Path(rec.video_path).name == "spatial_overlay.mp4":
        raise SystemExit("refusing spatial_overlay.mp4 as input")

    print(f"decoding {rec.video_path}", flush=True)
    raw, info = decode_all_rgb(rec.video_path)
    frames_u8 = preprocess_clip(raw, signature_from_cfg(cfg))
    heat = load_heatmaps(rec.heatmap_json_path, expected_frames=int(frames_u8.shape[0]))
    print(f"frames={frames_u8.shape} fps={info.fps} video={info.path}", flush=True)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    still_dir = out_dir / "stills"
    still_dir.mkdir(parents=True, exist_ok=True)
    fps = float(cfg.get("data.fps_expected", 30)) / max(int(args.stride), 1)
    summary = {"clip_id": rec.clip_id, "runs": {}}

    print("loading frozen Giant once", flush=True)
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    if int(getattr(model.backbone, "hidden_size", 0)) != 1408:
        raise SystemExit(f"need Giant C=1408, got {model.backbone.hidden_size}")
    model.to(device).eval()

    for item in args.checkpoints:
        name, path = item.split("=", 1)
        ckpt = Path(path)
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        print(f"=== {name} {ckpt} ===", flush=True)
        blob = load_checkpoint(ckpt, map_location=device)
        load_teacher_state(model, blob)
        frames, stills, rows = render_one(
            model, frames_u8, heat, rec, cfg, device, name, args.stride, args.batch_size
        )
        mp4 = out_dir / f"{name}_{rec.clip_id.replace('/', '__')}.mp4"
        _write_mp4(mp4, frames, fps)
        for t, panel in stills.items():
            Image.fromarray(panel).save(still_dir / f"{name}_t{t:03d}.png")
        n_stop = sum(1 for r in rows if r["label"] == "STOP")
        n_threat = sum(1 for r in rows if r["label"] == "THREAT")
        n_safe = sum(1 for r in rows if r["label"] == "SAFE")
        summary["runs"][name] = {
            "checkpoint": str(ckpt),
            "epoch": blob.get("epoch"),
            "mp4": str(mp4),
            "n_frames": len(frames),
            "n_SAFE": n_safe,
            "n_THREAT": n_threat,
            "n_STOP": n_stop,
            "mean_peak": float(np.mean([r["peak"] for r in rows])),
        }
        print(f"  wrote {mp4}  SAFE={n_safe} THREAT={n_threat} STOP={n_stop}", flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
