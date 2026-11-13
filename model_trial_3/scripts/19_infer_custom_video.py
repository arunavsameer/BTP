#!/usr/bin/env python3
"""Run a direct_h1s teacher checkpoint on an arbitrary RGB mp4.

Samples 16 frames by time over the previous 3.0 seconds (not by training-set
frame index). The street source is AV1, so frames come from ffmpeg at the
container fps. Overlay frames are 256x256 and are written as H.264.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.preprocessing import letterbox_rgb, uint8_to_nchw01
from collision_jepa3.evaluation.visualizations import overlay_heatmap
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state
from collision_jepa3.utils.device import resolve_device

OFFSETS_SEC = np.linspace(-3.0, 0.0, 16, dtype=np.float64)  # 0.2 s apart
CAUTION, STOP = 0.25, 0.60
ADAPTER_KEYS = (
    "projection_channels",
    "z_channels",
    "feature_grid",
    "temporal_blocks",
    "temporal_kernel",
    "temporal_dilation",
    "temporal_dilations",
    "temporal_pool",
    "group_norm_groups",
    "decoder_dilated",
    "decode_from",
    "heatmap_grid",
    "heatmap_pool",
)


def band(peak: float) -> str:
    if peak >= STOP:
        return "STOP"
    if peak >= CAUTION:
        return "THREAT"
    return "SAFE"


def hud_color(label: str) -> tuple[int, int, int]:
    return {"SAFE": (40, 180, 40), "THREAT": (0, 180, 255), "STOP": (0, 0, 255)}[label]


def _same(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return False
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return a == b


def _frame_rate(stream: dict) -> float:
    for key in ("r_frame_rate", "avg_frame_rate"):
        raw = str(stream.get(key) or "")
        if "/" not in raw:
            continue
        num, den = raw.split("/", 1)
        den_f = float(den)
        if den_f == 0.0:
            continue
        fps = float(num) / den_f
        if fps > 1e-3:
            return fps
    return 30.0


def probe_video(path: Path) -> tuple[int, int, float, float]:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        text=True,
    )
    info = json.loads(raw)
    stream = info["streams"][0]
    duration = float(info["format"]["duration"])
    return int(stream["width"]), int(stream["height"]), _frame_rate(stream), duration


def load_rgb_video(path: Path, start: float | None, end: float | None) -> tuple[np.ndarray, float, float]:
    """Decode [start, end] at the container fps. Returns RGB, fps, origin seconds.

    origin is the source time of frame 0. No fps resample and no center crop.
    """
    width, height, fps, file_duration = probe_video(path)
    origin = 0.0 if start is None else float(start)
    stop = file_duration if end is None else float(end)
    if stop <= origin:
        raise SystemExit(f"empty window {origin:.3f}s–{stop:.3f}s")
    seconds = stop - origin
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{origin:.6f}",
        "-i", str(path),
        "-t", f"{seconds:.6f}",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-an", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    frames = []
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy())
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr is not None else ""
        proc.wait()
    if not frames:
        raise SystemExit(f"ffmpeg returned no frames for {path} ({origin:.3f}s–{stop:.3f}s): {err.strip()}")
    return np.stack(frames, axis=0), fps, origin


def window_at(rgb: np.ndarray, fps: float, t_sec: float, origin: float) -> np.ndarray:
    n = len(rgb)
    idx = [int(np.clip(round((t_sec - origin + dt) * fps), 0, n - 1)) for dt in OFFSETS_SEC]
    clips = [letterbox_rgb(rgb[i], 256, 0) for i in idx]
    return uint8_to_nchw01(np.stack(clips, 0))  # [16,3,256,256] in [0,1]


def apply_checkpoint_adapter(cfg: Config, blob: dict) -> None:
    """Use the checkpoint adapter settings when they differ from default.yaml.

    heads.mode is forced to direct_h1s by the caller after this returns.
    """
    saved = blob.get("config") or {}
    saved_ad = saved.get("adapter") or {}
    for key in ADAPTER_KEYS:
        if key not in saved_ad:
            continue
        current = cfg.get(f"adapter.{key}")
        value = saved_ad[key]
        if not _same(current, value):
            print(
                f"adapter.{key}: default {current!r} -> checkpoint {value!r}",
                flush=True,
            )
            cfg.set(f"adapter.{key}", value)
    mid = blob.get("backbone_model_id") or (saved.get("backbone") or {}).get("model_id")
    if mid:
        # Same id as fallback so a failed Giant load cannot switch to ViT-L.
        cfg.set("backbone.model_id", str(mid))
        cfg.set("backbone.fallback_model_id", str(mid))


def open_h264_writer(path: Path, fps: float, size: int = 256) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.name == "street_vlog":
        raise SystemExit(f"refusing to overwrite an existing phase-A video: {path}")
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{size}x{size}", "-r", f"{fps:.8f}",
        "-i", "-",
        "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


def close_h264_writer(proc: subprocess.Popen, path: Path) -> None:
    assert proc.stdin is not None
    proc.stdin.close()
    err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr is not None else ""
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed writing {path}: {err.strip()}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True, help="output overlay mp4")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--device", default="auto")
    p.add_argument("--stride-seconds", type=float, default=0.2)
    p.add_argument("--hf-cache", default="", help="existing Giant cache, or empty to download")
    p.add_argument("--start", type=float, default=None, help="window start in source seconds")
    p.add_argument("--end", type=float, default=None, help="window end in source seconds")
    args = p.parse_args()

    if args.hf_cache:
        os.environ["HF_HOME"] = args.hf_cache

    blob = load_checkpoint(args.checkpoint, map_location="cpu")
    if blob.get("lora_enabled"):
        raise SystemExit("this checkpoint has LoRA; these three runs do not")

    cfg = Config.load(args.config)
    apply_checkpoint_adapter(cfg, blob)
    cfg.set("heads.mode", "direct_h1s")
    if args.hf_cache:
        cfg.set("paths.hf_cache_dir", args.hf_cache)
    else:
        cfg.set("paths.hf_cache_dir", "")
        cfg.set("paths.hf_cache_fallback", "")

    device = resolve_device(args.device)
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    if int(model.backbone.hidden_size) != 1408:
        raise SystemExit("need Giant (C=1408); got %s" % model.backbone.hidden_size)
    load_teacher_state(model, blob)
    model.to(device).eval()
    print(
        "loaded", args.checkpoint, "epoch", blob.get("epoch"),
        "backbone", getattr(model.backbone, "model_id", None),
        "hidden", model.backbone.hidden_size,
        flush=True,
    )

    rgb, fps, origin = load_rgb_video(Path(args.video), args.start, args.end)
    n, h0, w0 = rgb.shape[:3]
    span = (n - 1) / fps
    last_t = origin + span
    limit = last_t if args.end is None else min(float(args.end), last_t)
    if limit < origin + 3.05:
        raise SystemExit(f"video is {span:.2f}s; need ≥ 3.05s of history")
    print(
        f"video frames={n} fps={fps:.3f} size={w0}x{h0} dur={span:.2f}s origin={origin:.3f}s",
        flush=True,
    )

    out_path = Path(args.out)
    csv_path = out_path.with_suffix(".csv")
    out_fps = max(1.0, 1.0 / args.stride_seconds)
    writer = open_h264_writer(out_path, out_fps, 256)
    assert writer.stdin is not None
    t0 = origin + 3.0
    rows = []
    t = t0
    try:
        with torch.no_grad():
            while t <= limit + 1e-6:
                clip = torch.from_numpy(window_at(rgb, fps, t, origin))[None].to(device)
                if device.startswith("cuda"):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = model(clip=clip)
                else:
                    out = model(clip=clip)
                hm = out["Hhat_1s"][0].float().cpu().numpy()
                peak = float(hm.max())
                label = band(peak)
                anchor_i = int(np.clip(round((t - origin) * fps), 0, n - 1))
                vis = overlay_heatmap(letterbox_rgb(rgb[anchor_i], 256, 0), hm, alpha=0.45)
                vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
                cv2.rectangle(vis_bgr, (0, 0), (255, 28), hud_color(label), -1)
                cv2.putText(
                    vis_bgr,
                    f"{label}  peak={peak:.2f}  t={t:.2f}s  (+1s)",
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                writer.stdin.write(np.ascontiguousarray(vis_rgb, dtype=np.uint8).tobytes())
                rows.append({"t_sec": f"{t:.3f}", "peak": f"{peak:.4f}", "label": label})
                print(f"t={t:8.2f}s  {label:6s}  peak={peak:.3f}", flush=True)
                t += args.stride_seconds
    finally:
        close_h264_writer(writer, out_path)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["t_sec", "peak", "label"])
        w.writeheader()
        w.writerows(rows)
    counts = Counter(row["label"] for row in rows)
    print(
        f"counts SAFE={counts['SAFE']} THREAT={counts['THREAT']} STOP={counts['STOP']}",
        flush=True,
    )
    print("wrote", out_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
