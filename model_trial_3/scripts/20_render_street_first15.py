#!/usr/bin/env python3
"""Teacher H+1s overlay on the first 15 minutes of street_walking_480p.mp4.

Streams ffmpeg so the 15-minute window is not held in RAM. Layout matches
16_render_street_vlog.py (full-res overlay + 5×5 grid). First 3s have no
prediction (16 frames over the last 90).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.data.preprocessing import letterbox_rgb, uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import input_indices  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402
from scripts._common import add_config_args, load_cfg  # noqa: E402

spec_path = ROOT / "scripts" / "16_render_street_vlog.py"
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("street_vlog_teacher", spec_path)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)

SOURCE = Path("/storage/BTP/impl/input/street_walking_480p.mp4")
CKPT = Path("/storage/BTP/model_trial_3/checkpoints/teacher_phase_a_epoch_08.pt")
HF_CACHE = Path("/storage/BTP/model_trial_3/hf_cache")
OUT_DEFAULT = ROOT / "visualizations" / "street_first15min" / "street_walking_first15min.mp4"
WIDTH, HEIGHT, FPS = 854, 480, 30.0
OFFSETS = [-90, -84, -78, -72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]
HISTORY = 90
DURATION_SEC = 15 * 60
STOP_THRESHOLD = 0.60
OVERLAY_ALPHA = 0.45


def overlay_on_frame_binary(rgb: np.ndarray, hm: np.ndarray, size: int, stop_thr: float = STOP_THRESHOLD) -> np.ndarray:
    """Letterboxed overlay: STOP cells (>= thr) solid red, others solid blue. No blend of values."""
    from PIL import Image

    square = letterbox_rgb(rgb, size, 0)
    heat = np.clip(np.asarray(hm, dtype=np.float32), 0.0, 1.0)
    stop = heat >= float(stop_thr)
    heat_img = np.zeros(heat.shape + (3,), dtype=np.uint8)
    heat_img[stop] = (255, 0, 0)
    heat_img[~stop] = (0, 0, 255)
    heat_img = np.array(Image.fromarray(heat_img).resize((size, size), Image.NEAREST))
    painted = (
        square.astype(np.float32) * (1.0 - OVERLAY_ALPHA) + heat_img.astype(np.float32) * OVERLAY_ALPHA
    ).clip(0, 255).astype(np.uint8)
    x0, y0, new_w, new_h = _mod.content_box(rgb.shape[1], rgb.shape[0], size)
    crop = painted[y0 : y0 + new_h, x0 : x0 + new_w]
    return cv2.resize(crop, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def compose_binary(rgb: np.ndarray, hm: np.ndarray | None, size: int, label: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    if hm is None:
        return _mod.compose(rgb, None, size, label)
    left = overlay_on_frame_binary(rgb, hm, size)
    right = _mod.side_grid(hm, h)
    canvas = np.concatenate([left, right], axis=1)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def apply_checkpoint_config(cfg: Config, blob: dict) -> None:
    saved = blob.get("config") or {}
    for key, value in (saved.get("adapter") or {}).items():
        cfg.set(f"adapter.{key}", value)
    mid = blob.get("backbone_model_id") or (saved.get("backbone") or {}).get("model_id")
    if mid:
        cfg.set("backbone.model_id", str(mid))
        cfg.set("backbone.fallback_model_id", str(mid))
    cfg.set("heads.mode", "direct_h1s")


def load_teacher(cfg: Config, checkpoint: Path, device: str, hf_cache: str):
    os.environ["HF_HOME"] = hf_cache
    cfg.set("paths.hf_cache_dir", hf_cache)
    cfg.set("paths.hf_cache_fallback", hf_cache)
    blob = load_checkpoint(checkpoint, map_location="cpu")
    if blob.get("lora_enabled"):
        raise SystemExit(f"{checkpoint} has LoRA enabled")
    apply_checkpoint_config(cfg, blob)
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    hidden = int(model.backbone.hidden_size)
    if hidden != 1408:
        raise SystemExit(f"need Giant (C=1408); got {hidden}")
    load_teacher_state(model, blob)
    model.to(device).eval()
    print(
        "loaded",
        checkpoint,
        "epoch",
        blob.get("epoch"),
        "phase",
        blob.get("phase"),
        "backbone",
        getattr(model.backbone, "model_id", None),
        flush=True,
    )
    return model


def open_decoder(path: Path, start: float, duration: float) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(path),
            "-t",
            f"{duration:.3f}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-an",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )


def open_writer(path: Path, fps: float, w: int, h: int) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{w}x{h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )


@torch.no_grad()
def predict_batch(model, clips: list[np.ndarray], device: str) -> list[np.ndarray]:
    x = torch.from_numpy(np.stack(clips, axis=0)).to(device)
    if device.startswith("cuda"):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(clip=x)
    else:
        pred = model(clip=x)
    return list(pred["Hhat_1s"].float().cpu().numpy())


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default=str(CKPT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    parser.add_argument("--hf-cache", default=str(HF_CACHE))
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=DURATION_SEC)
    parser.add_argument("--max-frames", type=int, default=0, help="debug: stop after N decoded frames")
    args = parser.parse_args()

    cfg = load_cfg(args)
    device = resolve_device(args.device)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("cuda unavailable; using cpu", flush=True)
    model = load_teacher(cfg, Path(args.checkpoint), device, args.hf_cache)
    size = int(cfg.get("preprocessing.input_size", 256))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path

    dec = open_decoder(Path(args.source), args.start, args.duration)
    assert dec.stdout is not None
    writer = None
    frame_bytes = WIDTH * HEIGHT * 3
    history = deque(maxlen=HISTORY + 1)
    pending_clips: list[np.ndarray] = []
    pending_rgb: list[np.ndarray] = []
    pending_t: list[int] = []
    t = 0
    peak_all = 0.0
    n_pred = 0

    def flush_pending(force: bool = False) -> None:
        nonlocal peak_all, n_pred
        if not pending_clips:
            return
        if not force and len(pending_clips) < args.batch_size:
            return
        hms = predict_batch(model, pending_clips, device)
        for rgb, hm, idx in zip(pending_rgb, hms, pending_t):
            peak = float(hm.max())
            peak_all = max(peak_all, peak)
            n_pred += 1
            label = f"first15min  t={idx}  teacher ep08 H+1s  peak {peak:.2f}"
            canvas = compose_binary(rgb, hm, size, label)
            writer.stdin.write(np.ascontiguousarray(canvas, dtype=np.uint8).tobytes())
        pending_clips.clear()
        pending_rgb.clear()
        pending_t.clear()
        if n_pred and n_pred % 150 == 0:
            print(f"  predicted {n_pred} frames  peak {peak_all:.3f}", flush=True)

    try:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if writer is None:
                sample = _mod.compose(rgb, None, size, "init")
                h, w = sample.shape[:2]
                writer = open_writer(out_path, FPS, w, h)
                assert writer.stdin is not None
                print(f"writing {out_path}  canvas {w}x{h}", flush=True)
            lb = letterbox_rgb(rgb, size, 0)
            history.append(lb)
            if t < HISTORY:
                canvas = _mod.compose(rgb, None, size, f"first15min  t={t}  waiting for 3s history")
                writer.stdin.write(np.ascontiguousarray(canvas, dtype=np.uint8).tobytes())
            else:
                buf = np.stack(history, axis=0)
                idx = input_indices(HISTORY, OFFSETS)
                pending_clips.append(uint8_to_nchw01(buf[idx]))
                pending_rgb.append(rgb)
                pending_t.append(t)
                flush_pending(force=False)
            t += 1
            if args.max_frames and t >= args.max_frames:
                break
            if t % 900 == 0:
                print(f"decoded {t} frames ({t / FPS:.1f}s)", flush=True)
        flush_pending(force=True)
    finally:
        dec.stdout.close()
        dec.wait()
        if writer is not None:
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {out_path}")
    print(f"wrote {out_path}  frames={t}  predicted={n_pred}  peak={peak_all:.3f}", flush=True)


if __name__ == "__main__":
    main()
