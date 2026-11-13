#!/usr/bin/env python3
"""Teacher H+1s overlay on street-vlog windows.

The source is AV1, so frames come from ffmpeg. The first 3 seconds of each
clip have no prediction: the teacher reads 16 frames spaced over the last
90 frames. There is no ground-truth heatmap for this video.

--clips original writes the three windows already used by impl2.
--clips extra samples six more 15–20s windows (seed 42, 5s gap) that stay
clear of those three and of each other, and writes clip_04 onward.
--clips all writes those nine windows into --out-dir (not the phase-A folder).
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg, resolve_checkpoint  # noqa: E402

from collision_jepa3.data.preprocessing import letterbox_rgb, uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import input_indices  # noqa: E402
from collision_jepa3.evaluation.visualizations import heatmap_to_image, overlay_heatmap  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402

SOURCE = Path("/storage/BTP/impl/input/street_walking_480p.mp4")
# Same windows as impl2 (seed 0).
CLIPS = [
    ("clip_01_753s-770s.mp4", 752.5494252392793, 769.6522831434336),
    ("clip_02_1177s-1194s.mp4", 1176.7697669163908, 1194.3261405232338),
    ("clip_03_2201s-2221s.mp4", 2201.411274984765, 2220.63338424239),
]
# The three windows above plus the six already rendered as clip_04..clip_09.
ALL_CLIPS = CLIPS + [
    ("clip_04_73s-91s.mp4", 72.66715358845768, 90.8642875807471),
    ("clip_05_252s-272s.mp4", 252.4853612674675, 271.9462591059917),
    ("clip_06_578s-593s.mp4", 578.319436893243, 593.4521167416624),
    ("clip_07_649s-665s.mp4", 648.9312457773718, 665.3063923692174),
    ("clip_08_1469s-1485s.mp4", 1469.3407867015221, 1485.43397657554),
    ("clip_09_1966s-1984s.mp4", 1965.7788370766618, 1984.461193147482),
]
WIDTH = 854
HEIGHT = 480
FPS = 30.0
# Not seed 0: that seed is the three windows above.
EXTRA_SEED = 42
EXTRA_COUNT = 6
CLIP_SECONDS = (15.0, 20.0)
CLIP_GAP = 5.0


def probe_duration(path: Path) -> float:
    raw = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        text=True,
    )
    return float(raw.strip())


def clip_filename(index: int, start: float, end: float) -> str:
    return f"clip_{index:02d}_{round(start)}s-{round(end)}s.mp4"


def _too_close(start: float, end: float, others: list[tuple[float, float]], gap: float) -> bool:
    """True when [start, end] comes within `gap` seconds of any other window."""
    return any(start < old_end + gap and old_start < end + gap for old_start, old_end in others)


def sample_extra_clips(
    duration: float,
    blocked: list[tuple[float, float]],
    count: int = EXTRA_COUNT,
    seed: int = EXTRA_SEED,
    gap: float = CLIP_GAP,
) -> list[tuple[str, float, float]]:
    """Non-overlapping 15–20s windows, `gap` seconds clear of blocked and each other."""
    lo, hi = CLIP_SECONDS
    if duration <= hi:
        raise RuntimeError(f"video is too short to sample ({duration:.1f}s)")
    rng = random.Random(seed)
    chosen: list[tuple[float, float]] = []
    for _ in range(20000):
        if len(chosen) == count:
            break
        length = rng.uniform(lo, hi)
        start = rng.uniform(0.0, duration - length)
        end = start + length
        if not _too_close(start, end, blocked + chosen, gap):
            chosen.append((start, end))
    if len(chosen) != count:
        raise RuntimeError(f"only placed {len(chosen)} of {count} clips")
    chosen.sort()
    used = {name for name, _, _ in CLIPS}
    clips = []
    for index, (start, end) in enumerate(chosen, start=len(CLIPS) + 1):
        name = clip_filename(index, start, end)
        if name in used:
            raise RuntimeError(f"duplicate clip name {name}")
        used.add(name)
        clips.append((name, start, end))
    return clips


def decode_window(path: Path, start: float, end: float) -> np.ndarray:
    """Return [T,H,W,3] uint8 RGB."""
    seconds = end - start
    proc = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start:.3f}", "-i", str(path), "-t", f"{seconds:.3f}",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-an", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert proc.stdout is not None
    frame_bytes = WIDTH * HEIGHT * 3
    frames = []
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            bgr = np.frombuffer(raw, dtype=np.uint8).reshape((HEIGHT, WIDTH, 3))
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        proc.stdout.close()
        proc.wait()
    if not frames:
        raise RuntimeError(f"ffmpeg returned no frames for {start:.1f}s–{end:.1f}s")
    return np.stack(frames, axis=0)


def content_box(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    """Where letterbox_rgb placed the picture inside the square."""
    scale = min(size / float(width), size / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    x0 = (size - new_w) // 2
    y0 = (size - new_h) // 2
    return x0, y0, new_w, new_h


def overlay_on_frame(rgb: np.ndarray, hm: np.ndarray, size: int) -> np.ndarray:
    """Paint the 5×5 map on the letterboxed view, then crop back to the frame.

    The heatmap covers the whole square, including the black bars, so mapping
    it straight onto 854×480 would stretch those bars onto the sky and ground.
    """
    square = letterbox_rgb(rgb, size, 0)
    painted = overlay_heatmap(square, hm, alpha=0.45)
    x0, y0, new_w, new_h = content_box(rgb.shape[1], rgb.shape[0], size)
    crop = painted[y0 : y0 + new_h, x0 : x0 + new_w]
    return cv2.resize(crop, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def side_grid(hm: np.ndarray, height: int) -> np.ndarray:
    from PIL import Image

    cell = max(1, height // hm.shape[0])
    im = heatmap_to_image(hm, cell=cell)
    im = im.resize((height, height), Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def compose(rgb: np.ndarray, hm: np.ndarray | None, size: int, label: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    if hm is None:
        left = rgb
        right = np.full((h, h, 3), 24, dtype=np.uint8)
        cv2.putText(right, "needs 3s", (24, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(right, "history", (40, h // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
    else:
        left = overlay_on_frame(rgb, hm, size)
        right = side_grid(hm, h)
    canvas = np.concatenate([left, right], axis=1)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
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


@torch.no_grad()
def predict_clip(model, frames_u8: np.ndarray, letterboxed: np.ndarray, offsets: list[int], device: str, batch_size: int) -> dict[int, np.ndarray]:
    n = frames_u8.shape[0]
    t0 = -min(offsets)
    anchors = list(range(t0, n))
    amp = torch.bfloat16 if device.startswith("cuda") else None
    out: dict[int, np.ndarray] = {}
    for start in range(0, len(anchors), batch_size):
        chunk = anchors[start : start + batch_size]
        clips = []
        for t in chunk:
            idx = input_indices(t, offsets)
            window = letterboxed[idx]
            clips.append(uint8_to_nchw01(window))
        x = torch.from_numpy(np.stack(clips, axis=0)).to(device)
        if amp is not None:
            with torch.autocast(device_type="cuda", dtype=amp):
                pred = model(clip=x)
        else:
            pred = model(clip=x)
        h_1s = pred["Hhat_1s"].float().cpu().numpy()
        for i, t in enumerate(chunk):
            out[t] = h_1s[i]
        done = start + len(chunk)
        if done % 80 == 0 or done == len(anchors):
            print(f"  {done}/{len(anchors)} anchors  peak {float(h_1s.max()):.3f}", flush=True)
    return out


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


def apply_checkpoint_config(cfg, blob: dict) -> None:
    """Copy adapter/backbone settings that differ, then force direct_h1s."""
    saved = blob.get("config") or {}
    for key, value in (saved.get("adapter") or {}).items():
        current = cfg.get(f"adapter.{key}")
        if not _same(current, value):
            print(f"adapter.{key}: default {current!r} -> checkpoint {value!r}", flush=True)
            cfg.set(f"adapter.{key}", value)
    mid = blob.get("backbone_model_id") or (saved.get("backbone") or {}).get("model_id")
    if mid:
        cfg.set("backbone.model_id", str(mid))
        cfg.set("backbone.fallback_model_id", str(mid))
    cfg.set("heads.mode", "direct_h1s")


def load_teacher(cfg, checkpoint: Path, device: str, hf_cache: str):
    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        cfg.set("paths.hf_cache_dir", hf_cache)
    blob = load_checkpoint(checkpoint, map_location="cpu")
    if blob.get("lora_enabled"):
        raise SystemExit(f"{checkpoint} has LoRA enabled; these runs stay frozen")
    apply_checkpoint_config(cfg, blob)
    model = CollisionTeacher3(cfg, dummy_backbone=False)
    hidden = int(model.backbone.hidden_size)
    if hidden != 1408:
        raise SystemExit(f"need Giant (C=1408); got {hidden}")
    load_teacher_state(model, blob)
    model.to(device).eval()
    print(
        "loaded", checkpoint,
        "epoch", blob.get("epoch"),
        "phase", blob.get("phase"),
        "backbone", getattr(model.backbone, "model_id", None),
        "hidden", hidden,
        flush=True,
    )
    return model


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--out-dir", default=None, help="directory for mp4s; default is visualizations/street_vlog")
    parser.add_argument("--hf-cache", default="", help="Giant cache directory; also sets HF_HOME")
    parser.add_argument(
        "--clips",
        choices=("original", "extra", "all"),
        default="original",
        help="original: three windows. extra: six new windows. all: the nine existing windows.",
    )
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = resolve_device(args.device)
    source = Path(args.source)
    if args.clips == "original":
        clips = list(CLIPS)
    elif args.clips == "all":
        clips = list(ALL_CLIPS)
    else:
        duration = probe_duration(source)
        blocked = [(start, end) for _, start, end in CLIPS]
        clips = sample_extra_clips(duration, blocked)
        print(
            f"sampled {len(clips)} clips  seed={EXTRA_SEED}  "
            f"gap={CLIP_GAP:.1f}s  duration={duration:.6f}s",
            flush=True,
        )
        for name, start, end in clips:
            print(f"  {name}  start={start!r}  end={end!r}  length={end - start:.6f}", flush=True)

    checkpoint = resolve_checkpoint(cfg, args.checkpoint)
    model = load_teacher(cfg, checkpoint, device, args.hf_cache)
    size = int(cfg.get("preprocessing.input_size", 256))
    offsets = [int(o) for o in cfg.get("sampling.frame_offsets")]

    phase_a = (cfg.path("paths.visualizations_root") / "street_vlog").resolve()
    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
    else:
        out_dir = phase_a
    out_dir = out_dir.resolve()
    if args.clips == "all" and out_dir == phase_a:
        raise SystemExit(f"refusing to overwrite phase-A videos in {phase_a}")
    print(f"writing wide overlays to {out_dir}", flush=True)
    written = []
    for name, start, end in clips:
        print(f"decoding {name} {start:.1f}s–{end:.1f}s", flush=True)
        rgb = decode_window(source, start, end)
        print(f"  {rgb.shape[0]} frames, letterboxing", flush=True)
        letterboxed = np.stack([letterbox_rgb(rgb[i], size, 0) for i in range(rgb.shape[0])], axis=0)
        preds = predict_clip(model, rgb, letterboxed, offsets, device, args.batch_size)
        peak = max(float(hm.max()) for hm in preds.values())
        frames = []
        for t in range(rgb.shape[0]):
            hm = preds.get(t)
            if hm is None:
                label = f"{name}  t={t}  waiting for 3s history"
            else:
                label = f"{name}  t={t}  teacher H+1s  peak {float(hm.max()):.2f}"
            frames.append(compose(rgb[t], hm, size, label))
        path = out_dir / name
        write_mp4(path, frames, FPS)
        written.append((path, len(frames), peak))
        print(f"  wrote {path} ({len(frames)} frames, pred peak {peak:.3f})", flush=True)
    print("summary", flush=True)
    for path, nframes, peak in written:
        print(f"  {path}  frames={nframes}  peak={peak:.6f}", flush=True)


if __name__ == "__main__":
    main()
