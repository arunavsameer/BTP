"""Laptop collision demo.

Reads the 480p walking clip (the same file used as the stream source),
Runs YOLOE tracking, then draws collision forecasts from a linear fit
over the last k boxes (least squares on centre and size).

Tweak CollisionConfig in this file (or CLI flags) — you should not need to
edit collision.py for ordinary experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLOE

from collision import (
    Box,
    CollisionConfig,
    HitLatch,
    TrackMemory,
    draw_forecasts,
    draw_margin_frame,
    forecast_from_window,
)
from tracker import (
    CROPPED_DIR,
    MODEL_WEIGHTS,
    OUTPUT_DIR,
    TARGET_CLASSES,
    open_frame_source,
    weights_path,
)

DEFAULT_VIDEO = CROPPED_DIR / "untitledwalking.mp4"


def parse_args() -> argparse.Namespace:
    cfg = CollisionConfig()
    p = argparse.ArgumentParser(description="Detect objects and mark likely collisions.")
    p.add_argument("--source", default=str(DEFAULT_VIDEO), help="Input video (file only).")
    p.add_argument("--output", default=str(OUTPUT_DIR / "collision_output.mp4"))
    p.add_argument("--frame-gap", type=int, default=cfg.frame_gap, help="k: last k frames used in the linear fit.")
    p.add_argument(
        "--frame-margin-vertical",
        type=float,
        default=cfg.frame_margin_vertical,
        help="Vertical hit-zone scale. 1.0=screen height, 1.5=50%% taller.",
    )
    p.add_argument(
        "--frame-margin-horizontal",
        type=float,
        default=cfg.frame_margin_horizontal,
        help="Horizontal hit-zone scale. 1.0=screen width, 0.8=middle 80%%.",
    )
    p.add_argument(
        "--confirm-hits",
        type=int,
        default=cfg.confirm_hits,
        help="Mark HIT only after this many consecutive collision forecasts (1 = no wait).",
    )
    p.add_argument("--size-axis", default=cfg.size_axis, choices=["first_to_fill", "width", "height", "max"])
    p.add_argument("--min-growth-ratio", type=float, default=cfg.min_growth_ratio)
    p.add_argument("--max-ttc-seconds", type=float, default=cfg.max_ttc_seconds)
    p.add_argument("--conf", type=float, default=cfg.conf)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--classes",
        nargs="+",
        default=TARGET_CLASSES,
        help="Open-vocab class names passed to YOLOE (space-separated).",
    )
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> CollisionConfig:
    return CollisionConfig(
        frame_gap=args.frame_gap,
        frame_margin_vertical=args.frame_margin_vertical,
        frame_margin_horizontal=args.frame_margin_horizontal,
        size_axis=args.size_axis,
        min_growth_ratio=args.min_growth_ratio,
        max_ttc_seconds=args.max_ttc_seconds,
        conf=args.conf,
        confirm_hits=args.confirm_hits,
    )


def detections_from_result(result) -> list[tuple[Box, int, str, float]]:
    """Only objects with a track id can be forecast (we must match the same thing later)."""
    items = []
    if result.boxes is None or len(result.boxes) == 0 or result.boxes.id is None:
        return items
    boxes = result.boxes.xyxy.cpu().tolist()
    ids = result.boxes.id.int().cpu().tolist()
    classes = result.boxes.cls.int().cpu().tolist()
    confs = result.boxes.conf.cpu().tolist()
    names = result.names
    for xyxy, track_id, cls_idx, conf in zip(boxes, ids, classes, confs):
        items.append((Box.from_xyxy(xyxy), int(track_id), names[int(cls_idx)], float(conf)))
    return items


def draw_hud(frame, cfg: CollisionConfig, n_hit: int, n_tracks: int, frame_idx: int) -> None:
    lines = [
        f"frame {frame_idx}   tracks {n_tracks}   collisions {n_hit}",
        f"fit last k={cfg.frame_gap} frames (linreg)   "
        f"zone={cfg.frame_margin_horizontal:.2f}xW {cfg.frame_margin_vertical:.2f}xH   "
        f"confirm={cfg.confirm_hits}",
    ]
    y = 22
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        y += 20


def run(cfg: CollisionConfig, source: str, output: str, imgsz: int, classes: list[str] | None = None) -> None:
    classes = list(classes) if classes else list(TARGET_CLASSES)
    weights = weights_path()
    print(f"Loading {weights.name}...")
    model = YOLOE(str(weights) if weights.exists() else MODEL_WEIGHTS)
    model.eval()
    model.set_classes(classes)
    print(f"YOLOE class names after set_classes: {model.names}")

    cap, width, height, fps = open_frame_source(source)
    if cap is None:
        raise SystemExit(f"Could not open {source}")
    if fps is None or fps <= 1e-3:
        fps = 30.0
    writer_fps = int(round(fps))

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), writer_fps, (width, height))
    memory = TrackMemory(cfg.frame_gap)
    latch = HitLatch(cfg.confirm_hits)
    log = []
    frame_idx = 0

    print(f"Reading {source}")
    print(f"classes={classes}")
    print(
        f"k={cfg.frame_gap} (linear fit), "
        f"margin H={cfg.frame_margin_horizontal}x V={cfg.frame_margin_vertical}x, "
        f"confirm_hits={cfg.confirm_hits}, axis={cfg.size_axis}"
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        result = model.track(
            frame,
            persist=True,
            tracker="botsort.yaml",
            conf=cfg.conf,
            imgsz=imgsz,
            verbose=False,
        )[0]

        current = detections_from_result(result)
        forecasts = []
        for box, track_id, class_name, _conf in current:
            memory.update(frame_idx, track_id, box, class_name)
            samples = memory.window(track_id, frame_idx)
            if len(samples) < cfg.min_samples:
                continue
            forecasts.append(
                forecast_from_window(
                    samples,
                    frame_idx,
                    fps,
                    width,
                    height,
                    cfg,
                    track_id,
                    class_name,
                )
            )

        live_ids = {track_id for _box, track_id, _name, _conf in current}
        forecasts = latch.apply(forecasts, live_ids)

        vis = frame.copy()
        draw_margin_frame(vis, cfg.frame_margin_horizontal, cfg.frame_margin_vertical)
        draw_forecasts(vis, forecasts, [(b, i, n) for b, i, n, _ in current])
        hits = [f for f in forecasts if f.will_collide]
        draw_hud(vis, cfg, len(hits), len(current), frame_idx)
        writer.write(vis)

        for f in hits:
            log.append(
                {
                    "frame": frame_idx,
                    "track_id": f.track_id,
                    "class": f.class_name,
                    "ttc_seconds": f.ttc_seconds,
                    "impact_center": [f.impact_cx, f.impact_cy],
                    "reason": f.reason,
                }
            )

        if frame_idx % 30 == 0:
            print(f"  {frame_idx} frames, {len(hits)} collision flags this frame")

    cap.release()
    writer.release()
    log_path = Path(output).with_suffix(".json")
    log_path.write_text(
        json.dumps({"config": cfg.__dict__, "classes": classes, "collisions": log}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output}")
    print(f"Wrote {log_path} ({len(log)} collision flags)")


if __name__ == "__main__":
    args = parse_args()
    run(config_from_args(args), args.source, args.output, args.imgsz, args.classes)
