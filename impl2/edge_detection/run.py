"""Edge-region collision demo.

Same forecast as impl/run_collision.py and the color-blob demo: follow each
box for a few frames, fit a line to its centre and size, and mark a hit if
it is growing toward the camera and would still be in the hit zone when it
fills the view.

The boxes do not come from YOLO or from color. Edges are walls, and each
connected patch of non-edge pixels is one box. Change the thresholds in
config.py.

The source is AV1, which OpenCV on this machine does not decode, so frames
are read through ffmpeg. The three clips are the same windows as the
color-blob run (seed 0), so the videos can be compared side by side.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# collision.py is the YOLO demo's forecast. It does not depend on YOLO.
sys.path.insert(0, str(ROOT.parent.parent / "impl"))
sys.path.insert(0, str(ROOT))

from collision import (
    Box,
    CollisionConfig,
    HitLatch,
    TrackMemory,
    draw_forecasts,
    draw_margin_frame,
    forecast_from_window,
)
from config import Config, validate
from segment import find_regions
from tracker import RegionTracker

DEFAULT_SOURCE = ROOT.parent.parent / "impl" / "input" / "street_walking_480p.mp4"
OUTPUT_DIR = ROOT / "output"

# Same windows as impl2/output from the color-blob run, seed 0.
CLIPS = [
    ("clip_01_753s-770s.mp4", 752.5494252392793, 769.6522831434336),
    ("clip_02_1177s-1194s.mp4", 1176.7697669163908, 1194.3261405232338),
    ("clip_03_2201s-2221s.mp4", 2201.411274984765, 2220.63338424239),
]


def parse_args() -> argparse.Namespace:
    cfg = Config()
    hit = CollisionConfig()
    p = argparse.ArgumentParser(description="Collision forecast from edge regions instead of YOLO.")
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--blur", type=int, default=cfg.blur_ksize, help="Odd Gaussian kernel before Canny. 0 or 1 turns it off.")
    p.add_argument("--canny-low", type=float, default=cfg.canny_low)
    p.add_argument("--canny-high", type=float, default=cfg.canny_high)
    p.add_argument("--aperture", type=int, default=cfg.canny_aperture, help="Sobel aperture inside Canny: 3, 5, or 7.")
    p.add_argument("--dilate", type=int, default=cfg.edge_dilate, help="Grow edges by this many working-frame pixels. 0 turns it off.")
    p.add_argument("--connectivity", type=int, default=cfg.connectivity, choices=(4, 8))
    p.add_argument("--min-area", type=float, default=cfg.min_area, help="Ignore regions smaller than this many original-frame pixels.")
    p.add_argument("--max-area-frac", type=float, default=cfg.max_area_frac, help="Drop a region bigger than this fraction of the frame. 1 keeps all.")
    p.add_argument("--min-fill", type=float, default=cfg.min_fill, help="Fraction of the box that must be non-edge. 0 turns it off.")
    p.add_argument("--max-boxes", type=int, default=cfg.max_boxes)
    p.add_argument("--process-width", type=int, default=cfg.process_width, help="Segment on a frame this wide. 0 uses the full frame.")
    p.add_argument("--iou-match", type=float, default=cfg.iou_match)
    p.add_argument("--max-missed", type=int, default=cfg.max_missed)
    p.add_argument("--frame-gap", type=int, default=hit.frame_gap)
    p.add_argument("--confirm-hits", type=int, default=hit.confirm_hits)
    p.add_argument("--min-growth-ratio", type=float, default=hit.min_growth_ratio)
    p.add_argument("--max-ttc-seconds", type=float, default=hit.max_ttc_seconds)
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> tuple[Config, CollisionConfig]:
    cfg = Config(
        blur_ksize=args.blur,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        canny_aperture=args.aperture,
        edge_dilate=args.dilate,
        connectivity=args.connectivity,
        min_area=args.min_area,
        max_area_frac=args.max_area_frac,
        min_fill=args.min_fill,
        max_boxes=args.max_boxes,
        process_width=args.process_width,
        iou_match=args.iou_match,
        max_missed=args.max_missed,
    )
    hit = CollisionConfig(
        frame_gap=args.frame_gap,
        confirm_hits=args.confirm_hits,
        min_growth_ratio=args.min_growth_ratio,
        max_ttc_seconds=args.max_ttc_seconds,
    )
    return cfg, hit


def probe_video(path: Path) -> tuple[int, int, float, float]:
    raw = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    info = json.loads(raw)
    stream = next(s for s in info["streams"] if "width" in s)
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    return int(stream["width"]), int(stream["height"]), fps, float(info["format"]["duration"])


def iter_frames(path: Path, start: float, seconds: float, width: int, height: int):
    """Decode one clip to BGR frames. -ss before -i, so the cut may land on a keyframe."""
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
            str(path),
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
    frame_bytes = width * height * 3
    assert proc.stdout is not None
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if len(raw) < frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
    finally:
        proc.stdout.close()
        proc.wait(timeout=10)


def draw_hud(frame, cfg: Config, source_t: float, n_hit: int, n_boxes: int, frame_idx: int) -> None:
    width_label = "full" if cfg.process_width <= 0 else f"{cfg.process_width}px"
    lines = [
        f"edges   t={source_t:6.1f}s   frame {frame_idx}   boxes {n_boxes}   collisions {n_hit}",
        (
            f"canny {cfg.canny_low:.0f}/{cfg.canny_high:.0f}   aperture {cfg.canny_aperture}   "
            f"blur {cfg.blur_ksize}   dilate {cfg.edge_dilate}px   conn {cfg.connectivity}"
        ),
        (
            f"min area {cfg.min_area:.0f}px   fill {cfg.min_fill:.2f}   "
            f"max area {cfg.max_area_frac:.0%}   process {width_label}"
        ),
    ]
    y = 20
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1)
        y += 18


def process_clip(
    source: Path,
    output: Path,
    start: float,
    end: float,
    width: int,
    height: int,
    fps: float,
    cfg: Config,
    hit: CollisionConfig,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(round(fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    tracker = RegionTracker(cfg.iou_match, cfg.max_missed)
    memory = TrackMemory(hit.frame_gap)
    latch = HitLatch(hit.confirm_hits)
    frame_idx = 0
    hit_flags = 0
    hit_tracks: set[int] = set()

    try:
        for frame in iter_frames(source, start, end - start, width, height):
            frame_idx += 1
            regions = find_regions(frame, cfg)
            tracked = tracker.update(regions)

            current = []
            forecasts = []
            for region, track_id in tracked:
                box = Box(region.x1, region.y1, region.x2, region.y2)
                current.append((box, track_id, region.name))
                memory.update(frame_idx, track_id, box, region.name)
                samples = memory.window(track_id, frame_idx)
                if len(samples) < hit.min_samples:
                    continue
                forecasts.append(
                    forecast_from_window(
                        samples, frame_idx, fps, width, height, hit, track_id, region.name
                    )
                )

            live_ids = {track_id for _box, track_id, _name in current}
            forecasts = latch.apply(forecasts, live_ids)

            vis = frame.copy()
            draw_margin_frame(vis, hit.frame_margin_horizontal, hit.frame_margin_vertical)
            draw_forecasts(vis, forecasts, current)
            hits = [forecast for forecast in forecasts if forecast.will_collide]
            draw_hud(vis, cfg, start + frame_idx / fps, len(hits), len(current), frame_idx)
            writer.write(vis)

            for forecast in hits:
                hit_flags += 1
                hit_tracks.add(forecast.track_id)

            if frame_idx % 30 == 0:
                print(f"    {frame_idx} frames, {len(current)} boxes, {len(hits)} hits", flush=True)
    finally:
        writer.release()

    if frame_idx == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg returned no frames for {start:.1f}s–{end:.1f}s")

    return {"frames": frame_idx, "hit_flags": hit_flags, "hit_tracks": sorted(hit_tracks)}


def main() -> None:
    args = parse_args()
    cfg, hit = config_from_args(args)
    validate(cfg)

    source = Path(args.source)
    out_dir = Path(args.output_dir)
    if not source.exists():
        raise SystemExit(f"missing video: {source}")

    width, height, fps, duration = probe_video(source)
    print(f"source {source}", flush=True)
    print(f"{width}x{height} @ {fps:.2f} fps, {duration:.1f}s", flush=True)
    print(
        f"canny={cfg.canny_low:.0f}/{cfg.canny_high:.0f} aperture={cfg.canny_aperture} "
        f"blur={cfg.blur_ksize} dilate={cfg.edge_dilate} conn={cfg.connectivity} "
        f"min_area={cfg.min_area:.0f}px min_fill={cfg.min_fill:.2f} "
        f"max_area={cfg.max_area_frac:.0%} boxes={cfg.max_boxes} width={cfg.process_width}",
        flush=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, start, end in CLIPS:
        if start < 0 or end > duration + 0.05:
            raise SystemExit(f"{name} is outside the source ({duration:.1f}s)")
        output = out_dir / name
        print(f"rendering {output.name}  {start:.1f}s – {end:.1f}s ({end - start:.1f}s)", flush=True)
        result = process_clip(source, output, start, end, width, height, fps, cfg, hit)
        payload = {
            "source": str(source),
            "output": str(output),
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": result["frames"],
            "hit_flags": result["hit_flags"],
            "hit_tracks": result["hit_tracks"],
            "config": cfg.__dict__,
            "collision": hit.__dict__,
        }
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"  wrote {output.name} ({result['frames']} frames, {result['hit_flags']} hit flags)",
            flush=True,
        )
        summary.append(
            {
                "file": output.name,
                "start_seconds": start,
                "end_seconds": end,
                "frames": result["frames"],
                "hit_flags": result["hit_flags"],
                "hit_tracks": result["hit_tracks"],
            }
        )

    (out_dir / "clips.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done. files are in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
