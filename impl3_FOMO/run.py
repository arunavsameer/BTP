"""Train FOMO on street centroids, then render the three demo clips.

The collision fit is the same one as the color-blob demo: a line through
center and size, and a hit when the box grows toward the camera and stays
in the hit zone.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "impl"))
sys.path.insert(0, str(ROOT))

from collision import (  # noqa: E402
    Box,
    CollisionConfig,
    HitLatch,
    TrackMemory,
    draw_forecasts,
    draw_margin_frame,
    forecast_from_window,
)
from config import CLASSES, Config  # noqa: E402
from dataset import build_cache, train_head  # noqa: E402
from geom import letterbox  # noqa: E402
from model import Fomo  # noqa: E402
from post import HeatmapGate, class_prob, detections_from_logits  # noqa: E402
from tracker import CentroidTracker  # noqa: E402

CLASS_COLORS = {
    "person": (60, 180, 75),
    "pole": (180, 180, 40),
    "fire hydrant": (40, 40, 220),
    "dustbin": (40, 140, 220),
    "cone": (0, 140, 255),
    "bicycle": (200, 80, 200),
}


def load_fomo(cfg: Config) -> tuple[Fomo, float]:
    model = Fomo(num_classes=len(CLASSES), input_size=cfg.input_size, torch_home=cfg.torch_home)
    checkpoint = torch.load(Path(cfg.cache) / "fomo_v3.pt", weights_only=False)
    model.head.load_state_dict(checkpoint["head"])
    model.eval()
    return model, float(checkpoint["threshold"])


def iter_clip(path: str, start: float, seconds: float, width: int = 854, height: int = 480):
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            path,
            "-t",
            f"{seconds:.3f}",
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
    assert proc.stdout is not None
    frame_bytes = width * height * 3
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()
    finally:
        proc.stdout.close()
        proc.wait(timeout=10)


def detect(model: Fomo, frame: np.ndarray, cfg: Config, threshold: float, gate: HeatmapGate) -> list[dict]:
    canvas, scale, pad_x, pad_y = letterbox(frame, cfg.input_size)
    tensor = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        logits = model(tensor)[0]
    prob = class_prob(logits[:-2])
    _state, active, cls_map = gate.update(prob)
    return detections_from_logits(
        logits,
        CLASSES,
        threshold,
        scale,
        pad_x,
        pad_y,
        cfg.stride,
        has_size=True,
        active=active,
        cls_map=cls_map,
    )


def draw_centroids(frame, dets: list[dict]) -> None:
    for det in dets:
        color = CLASS_COLORS.get(det["class_name"], (255, 255, 255))
        center = (int(round(det["cx"])), int(round(det["cy"])))
        cv2.circle(frame, center, 5, color, -1)
        cv2.circle(frame, center, 8, color, 1)


def process_clip(cfg: Config, model: Fomo, threshold: float, name: str, start: float, end: float) -> dict:
    out = Path(cfg.output) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps = 854, 480, 30.0
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    hit_cfg = CollisionConfig()
    tracker = CentroidTracker(cfg.match_px, cfg.max_missed, cfg.size_max_ratio, cfg.size_blend)
    # Stay lit only just below the on-threshold, so a ghost cell drops quickly.
    off = max(cfg.conf_off, threshold - 0.06)
    gate = HeatmapGate(cfg.heatmap_momentum, threshold, off)
    memory = TrackMemory(hit_cfg.frame_gap)
    latch = HitLatch(hit_cfg.confirm_hits)
    collisions = []
    frame_idx = 0
    for frame in iter_clip(cfg.video, start, end - start):
        frame_idx += 1
        dets = tracker.update(detect(model, frame, cfg, threshold, gate))
        current = []
        forecasts = []
        for det in dets:
            box = Box(det["x1"], det["y1"], det["x2"], det["y2"])
            label = det["class_name"]
            current.append((box, det["track_id"], label))
            memory.update(frame_idx, det["track_id"], box, label)
            samples = memory.window(det["track_id"], frame_idx)
            if len(samples) < hit_cfg.min_samples:
                continue
            forecasts.append(
                forecast_from_window(
                    samples, frame_idx, fps, width, height, hit_cfg, det["track_id"], label
                )
            )
        live = {det["track_id"] for det in dets}
        forecasts = latch.apply(forecasts, live)
        vis = frame.copy()
        draw_margin_frame(vis, hit_cfg.frame_margin_horizontal, hit_cfg.frame_margin_vertical)
        draw_forecasts(vis, forecasts, current)
        draw_centroids(vis, dets)
        hits = [f for f in forecasts if f.will_collide]
        hud = f"FOMO  frame {frame_idx}  objects {len(dets)}  hits {len(hits)}  conf {threshold:.2f}"
        cv2.putText(vis, hud, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3)
        cv2.putText(vis, hud, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        writer.write(vis)
        for forecast in hits:
            collisions.append(
                {
                    "frame": frame_idx,
                    "track_id": forecast.track_id,
                    "class": forecast.class_name,
                    "ttc_seconds": forecast.ttc_seconds,
                    "reason": forecast.reason,
                }
            )
        if frame_idx % 30 == 0:
            print(f"    {frame_idx} frames, {len(dets)} objects, {len(hits)} hits")
    writer.release()
    return {"frames": frame_idx, "collisions": collisions}


def main() -> None:
    cfg = Config()
    Path(cfg.cache).mkdir(parents=True, exist_ok=True)
    grids = build_cache(cfg)
    checkpoint = Path(cfg.cache) / "fomo_v3.pt"
    if not checkpoint.exists():
        train_head(cfg, grids)
    model, threshold = load_fomo(cfg)
    summary = []
    for name, start, end in cfg.clips:
        print(f"rendering {name}")
        result = process_clip(cfg, model, threshold, name, start, end)
        payload = {
            "clip": name,
            "start": start,
            "end": end,
            "frames": result["frames"],
            "threshold": threshold,
            "classes": CLASSES,
            "hits": len(result["collisions"]),
            "collisions": result["collisions"],
        }
        (Path(cfg.output) / name).with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  {result['frames']} frames, {len(result['collisions'])} hits")
        summary.append({"clip": name, "frames": result["frames"], "hits": len(result["collisions"])})
    (Path(cfg.output) / "clips.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done. videos in {cfg.output}")


if __name__ == "__main__":
    main()
