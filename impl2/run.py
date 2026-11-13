"""Color-blob collision demo.

Same idea as impl/run_collision.py: follow each box for a few frames, fit a
line to its centre and size, and mark a hit if it is growing toward the
camera and would still be in the hit zone when it fills the view.

The boxes do not come from YOLO. They come from flat-ish patches of similar
color (a shirt, a face, a door). Change the thresholds in config.py.

The source is AV1, which OpenCV on this machine does not decode, so frames
are read through ffmpeg.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
# collision.py is the YOLO demo's forecast. It does not depend on YOLO.
sys.path.insert(0, str(ROOT.parent / "impl"))
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
from motion import background_ids, track_motion
from segment import find_blobs
from tracker import BlobTracker

DEFAULT_SOURCE = ROOT.parent / "impl" / "input" / "street_walking_480p.mp4"
OUTPUT_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    cfg = Config()
    hit = CollisionConfig()
    p = argparse.ArgumentParser(description="Collision forecast from color blobs instead of YOLO.")
    p.add_argument("--source", default=str(DEFAULT_SOURCE))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--num-clips", type=int, default=cfg.num_clips)
    p.add_argument("--clip-min", type=float, default=cfg.clip_seconds_min)
    p.add_argument("--clip-max", type=float, default=cfg.clip_seconds_max)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument(
        "--color-tolerance",
        type=float,
        default=cfg.color_tolerance,
        help="0 = exact RGB only, 1 = any color joins the blob.",
    )
    p.add_argument("--min-area", type=float, default=cfg.min_area, help="Ignore blobs smaller than this many pixels.")
    p.add_argument("--max-area-frac", type=float, default=cfg.max_area_frac)
    p.add_argument("--min-fill", type=float, default=cfg.min_fill)
    p.add_argument("--max-boxes", type=int, default=cfg.max_boxes)
    p.add_argument("--process-width", type=int, default=cfg.process_width)
    p.add_argument("--blur", type=int, default=cfg.blur_ksize)
    p.add_argument(
        "--neighbor-compare",
        action="store_true",
        help="Compare each pixel to its neighbor instead of the seed color.",
    )
    p.add_argument("--iou-match", type=float, default=cfg.iou_match)
    p.add_argument("--frame-gap", type=int, default=hit.frame_gap)
    p.add_argument("--confirm-hits", type=int, default=hit.confirm_hits)
    p.add_argument("--min-growth-ratio", type=float, default=hit.min_growth_ratio)
    p.add_argument("--max-ttc-seconds", type=float, default=hit.max_ttc_seconds)
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> tuple[Config, CollisionConfig]:
    cfg = Config(
        num_clips=args.num_clips,
        clip_seconds_min=args.clip_min,
        clip_seconds_max=args.clip_max,
        seed=args.seed,
        color_tolerance=args.color_tolerance,
        min_area=args.min_area,
        max_area_frac=args.max_area_frac,
        min_fill=args.min_fill,
        max_boxes=args.max_boxes,
        process_width=args.process_width,
        blur_ksize=args.blur,
        compare_to_seed=not args.neighbor_compare,
        iou_match=args.iou_match,
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


def choose_clips(duration: float, cfg: Config, rng: random.Random) -> list[tuple[float, float]]:
    """Non-overlapping (start, end) times, each between clip_seconds_min and max."""
    if duration < 1.0:
        return []
    if duration <= cfg.clip_seconds_min:
        return [(0.0, duration)]

    length_hi = min(cfg.clip_seconds_max, duration)
    length_lo = min(cfg.clip_seconds_min, length_hi)
    fit = max(1, int(duration // length_lo))
    wanted = min(cfg.num_clips, fit)

    chosen = []
    for _ in range(4000):
        if len(chosen) == wanted:
            break
        length = rng.uniform(length_lo, length_hi)
        start = rng.uniform(0.0, duration - length)
        end = start + length
        if all(end <= old_start or start >= old_end for old_start, old_end in chosen):
            chosen.append((start, end))
    return sorted(chosen)


def iter_frames(path: Path, start: float, seconds: float, width: int, height: int):
    """Decode one clip to BGR frames. Fast seek, so the cut may land on a keyframe."""
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


def _area_ratio(samples) -> float:
    first = samples[0][1].area
    last = samples[-1][1].area
    if first <= 1.0:
        return 1.0
    return last / first


def _size_jitter(samples) -> float:
    """Average miss of the area from the straight line between the first and last sample."""
    areas = [sample[1].area for sample in samples]
    count = len(areas)
    if count < 3:
        return 0.0
    first, last = areas[0], areas[-1]
    error = 0.0
    for index, area in enumerate(areas):
        predicted = first + (last - first) * (index / (count - 1))
        error += abs(area - predicted)
    return (error / count) / max(first, last, 1.0)


def _passes_gates(samples, cfg: Config, hit: CollisionConfig, scene_ratio: float | None) -> bool:
    if cfg.require_both_axes:
        first, last = samples[0][1], samples[-1][1]
        if last.width < first.width * hit.min_growth_ratio or last.height < first.height * hit.min_growth_ratio:
            return False
    if cfg.max_size_jitter > 0 and _size_jitter(samples) > cfg.max_size_jitter:
        return False
    if scene_ratio is not None and _area_ratio(samples) < scene_ratio * cfg.scene_growth_margin:
        return False
    return True


def _metric_label(cfg: Config) -> str:
    if cfg.color_metric == "euclidean":
        radius = cfg.color_tolerance * 255.0 * (3.0 ** 0.5)
        return f"euclidean r<{radius:.0f}"
    return f"channel ±{round(cfg.color_tolerance * 255)}"


def draw_hud(frame, cfg: Config, source_t: float, n_hit: int, n_blobs: int, frame_idx: int, label: str = "") -> None:
    lines = []
    if label:
        lines.append(label)
    lines.append(f"t={source_t:6.1f}s   frame {frame_idx}   blobs {n_blobs}   collisions {n_hit}")
    lines.append(
        f"{_metric_label(cfg)}   min area {cfg.min_area:.0f}px   "
        f"fill {cfg.min_fill:.2f}   max area {cfg.max_area_frac:.0%}"
    )
    y = 22
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        y += 20


def draw_plain_boxes(frame, boxes) -> None:
    for box, track_id, name in boxes:
        p1 = (int(box.x1), int(box.y1))
        p2 = (int(box.x2), int(box.y2))
        cv2.rectangle(frame, p1, p2, (0, 200, 255), 2)
        cv2.putText(
            frame,
            f"{name} #{track_id}",
            (p1[0], max(16, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2,
        )


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
    label: str = "",
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        int(round(fps)),
        (width, height),
    )
    tracker = BlobTracker(
        iou_match=cfg.iou_match,
        max_missed=cfg.max_missed,
        min_track_frames=cfg.min_track_frames,
        smooth_frames=cfg.smooth_frames,
        center_match_frac=cfg.center_match_frac,
        center_match_px=cfg.center_match_px,
        size_ratio=cfg.size_ratio,
        color_absorb=cfg.color_absorb,
        absorb_iou=cfg.absorb_iou,
        memory_frames=cfg.memory_frames,
        memory_color=cfg.memory_color,
        memory_radius_px=cfg.memory_radius_px,
        memory_radius_cap=cfg.memory_radius_cap,
        memory_size_ratio=cfg.memory_size_ratio,
    )
    memory = TrackMemory(hit.frame_gap)
    latch = HitLatch(hit.confirm_hits)
    collisions = []
    frame_idx = 0

    for frame in iter_frames(source, start, end - start, width, height):
        frame_idx += 1
        blobs = find_blobs(frame, cfg)
        if cfg.drop_top_edge:
            blobs = [blob for blob in blobs if blob.y1 > cfg.top_edge_px]
        tracked = tracker.update(blobs)

        rows = []
        motions: dict[int, tuple[float, float, float]] = {}
        grace = hit.frame_gap
        for blob, track_id in tracked:
            box = Box(blob.x1, blob.y1, blob.x2, blob.y2)
            memory.update(frame_idx, track_id, box, blob.name)
            samples = memory.window(track_id, frame_idx)
            rows.append((box, track_id, blob.name, samples))
            span = samples[-1][0] - samples[0][0] if samples else 0
            if len(samples) >= 5 and span >= max(1, grace - 1):
                motion = track_motion(samples)
                if motion is not None:
                    motions[track_id] = motion

        background = background_ids(
            motions,
            outlier_mad=cfg.outlier_mad,
            floor=cfg.motion_floor,
            min_tracks=cfg.motion_min_tracks,
        )
        kept = [row for row in rows if row[1] not in background]

        current = [(box, track_id, name) for box, track_id, name, _samples in kept]
        pending = [
            (track_id, name, samples)
            for _box, track_id, name, samples in kept
            if len(samples) >= hit.min_samples and track_id in motions
        ]

        scene_ratio = None
        if cfg.use_scene_growth and pending:
            ratios = sorted(_area_ratio(samples) for _tid, _name, samples in pending)
            scene_ratio = ratios[len(ratios) // 2]

        forecasts = []
        for track_id, name, samples in pending:
            if not _passes_gates(samples, cfg, hit, scene_ratio):
                continue
            forecasts.append(
                forecast_from_window(samples, frame_idx, fps, width, height, hit, track_id, name)
            )

        live_ids = {track_id for _box, track_id, _name in current}
        forecasts = latch.apply(forecasts, live_ids)

        vis = frame.copy()
        if cfg.quiet_draw:
            hot_ids = {forecast.track_id for forecast in forecasts if forecast.will_collide or forecast.raw_collide}
            draw_plain_boxes(vis, [item for item in current if item[1] not in hot_ids])
            draw_forecasts(
                vis,
                [forecast for forecast in forecasts if forecast.track_id in hot_ids],
                [item for item in current if item[1] in hot_ids],
            )
        else:
            draw_margin_frame(vis, hit.frame_margin_horizontal, hit.frame_margin_vertical)
            draw_forecasts(vis, forecasts, current)
        hits = [forecast for forecast in forecasts if forecast.will_collide]
        draw_hud(vis, cfg, start + frame_idx / fps, len(hits), len(current), frame_idx, label)
        writer.write(vis)

        for forecast in hits:
            collisions.append(
                {
                    "frame": frame_idx,
                    "source_time": round(start + frame_idx / fps, 3),
                    "track_id": forecast.track_id,
                    "name": forecast.class_name,
                    "ttc_seconds": forecast.ttc_seconds,
                    "impact_center": [forecast.impact_cx, forecast.impact_cy],
                    "reason": forecast.reason,
                }
            )

        if frame_idx % 30 == 0:
            print(f"    {frame_idx} frames, {len(current)} blobs, {len(hits)} hits")

    writer.release()
    if frame_idx == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg returned no frames for {start:.1f}s–{end:.1f}s")

    return {"frames": frame_idx, "collisions": collisions}


def main() -> None:
    args = parse_args()
    cfg, hit = config_from_args(args)
    validate(cfg)

    source = Path(args.source)
    out_dir = Path(args.output_dir)
    if not source.exists():
        raise SystemExit(f"missing video: {source}")

    width, height, fps, duration = probe_video(source)
    rng = random.Random(cfg.seed)
    clips = choose_clips(duration, cfg, rng)
    if not clips:
        raise SystemExit("video is too short to cut a clip")

    print(f"source {source}")
    print(f"{width}x{height} @ {fps:.2f} fps, {duration:.1f}s")
    print(
        f"tolerance={cfg.color_tolerance:.2f} (channel gap {round(cfg.color_tolerance * 255)})  "
        f"min_area={cfg.min_area:.0f}px  min_fill={cfg.min_fill:.2f}  "
        f"max_area={cfg.max_area_frac:.0%}  seed={cfg.seed}"
    )
    print(f"{len(clips)} clip(s):")
    for index, (start, end) in enumerate(clips, start=1):
        print(f"  {index}. {start:.1f}s – {end:.1f}s ({end - start:.1f}s)")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for index, (start, end) in enumerate(clips, start=1):
        name = f"clip_{index:02d}_{start:.0f}s-{end:.0f}s.mp4"
        output = out_dir / name
        print(f"rendering {output.name}")
        result = process_clip(source, output, start, end, width, height, fps, cfg, hit)
        payload = {
            "source": str(source),
            "output": str(output),
            "start_seconds": start,
            "end_seconds": end,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": result["frames"],
            "config": cfg.__dict__,
            "collision": hit.__dict__,
            "collisions": result["collisions"],
        }
        log_path = output.with_suffix(".json")
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  wrote {output.name} ({result['frames']} frames, {len(result['collisions'])} hit flags)")
        summary.append({"clip": output.name, "start": start, "end": end, "hits": len(result["collisions"])})

    (out_dir / "clips.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"done. files are in {out_dir}")


if __name__ == "__main__":
    main()
