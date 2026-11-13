"""Render the three cheap box detectors on the same street clips.

Each method writes boxes, the overlap tracker follows them, and the
collision forecast in impl/collision.py decides the hit. The source is
AV1, which OpenCV on this machine does not decode, so frames are read
through ffmpeg the same way as impl2/run.py.

    python run.py
    python run.py --only pavement_mask --trap-top 0.72
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
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
from config import (
    FelzenszwalbConfig,
    FlowConfig,
    PavementConfig,
    TrackConfig,
    validate_felzenszwalb,
    validate_flow,
    validate_pavement,
    validate_track,
)
from felzenszwalb import find_regions as find_felzenszwalb
from felzenszwalb import segmentation_backend
from flow import FlowBoxes
from pavement import find_regions as find_pavement
from pavement import trapezoid_polygon
from regions import RegionTracker

DEFAULT_SOURCE = ROOT.parent.parent / "impl" / "input" / "street_walking_480p.mp4"
OUTPUT_DIR = ROOT / "output"

# Same windows as impl2/run_strategies.py (seed 0).
CLIPS = [
    ("clip_01_753s-770s.mp4", 752.5494252392793, 769.6522831434336),
    ("clip_02_1177s-1194s.mp4", 1176.7697669163908, 1194.3261405232338),
    ("clip_03_2201s-2221s.mp4", 2201.411274984765, 2220.63338424239),
]

METHODS = ("felzenszwalb", "pavement_mask", "sparse_flow")

ABOUT = {
    "felzenszwalb": (
        "Felzenszwalb-Huttenlocher graph segmentation (IJCV 2004): each region is a box, "
        "then the same overlap tracker and collision forecast as the YOLO demo."
    ),
    "pavement_mask": (
        "Felzenszwalb regions with Ulrich and Nourbakhsh (AAAI 2000) rejection: "
        "a box is dropped when its interior matches the hue/intensity histogram of the bottom road trapezoid."
    ),
    "sparse_flow": (
        "Lucas-Kanade corners with a partial-affine camera model removed; "
        "one box around each cluster of points that are moving apart, then the same tracker and collision forecast."
    ),
}


def parse_args() -> argparse.Namespace:
    felz = FelzenszwalbConfig()
    pav = PavementConfig()
    flow = FlowConfig()
    track = TrackConfig()
    hit = CollisionConfig()
    parser = argparse.ArgumentParser(
        description="Collision forecast from Felzenszwalb, a pavement mask, or sparse flow."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--only", default="all", choices=("all",) + METHODS)
    parser.add_argument("--iou-match", type=float, default=track.iou_match)
    parser.add_argument("--max-missed", type=int, default=track.max_missed)
    parser.add_argument("--frame-gap", type=int, default=hit.frame_gap)
    parser.add_argument("--confirm-hits", type=int, default=hit.confirm_hits)
    parser.add_argument("--min-growth-ratio", type=float, default=hit.min_growth_ratio)
    parser.add_argument("--max-ttc-seconds", type=float, default=hit.max_ttc_seconds)
    parser.add_argument("--k", type=float, default=felz.k, help="Felzenszwalb scale. Also used by pavement_mask.")
    parser.add_argument("--min-size", type=int, default=felz.min_size)
    parser.add_argument("--sigma", type=float, default=felz.sigma)
    parser.add_argument("--min-area", type=float, default=felz.min_area, help="Segment methods, full-frame pixels.")
    parser.add_argument("--max-area-frac", type=float, default=felz.max_area_frac)
    parser.add_argument("--min-fill", type=float, default=felz.min_fill)
    parser.add_argument("--max-boxes", type=int, default=felz.max_boxes)
    parser.add_argument("--process-width", type=int, default=felz.process_width)
    parser.add_argument("--trap-top", type=float, default=pav.trap_top)
    parser.add_argument("--trap-top-width", type=float, default=pav.trap_top_width)
    parser.add_argument("--hist-bins", type=int, default=pav.hist_bins)
    parser.add_argument("--match-threshold", type=float, default=pav.match_threshold)
    parser.add_argument("--reject-fraction", type=float, default=pav.reject_fraction)
    parser.add_argument("--gray-saturation", type=int, default=pav.gray_saturation)
    parser.add_argument("--max-corners", type=int, default=flow.max_corners)
    parser.add_argument("--quality", type=float, default=flow.quality)
    parser.add_argument("--min-distance", type=float, default=flow.min_distance)
    parser.add_argument("--lk-window", type=int, default=flow.lk_window)
    parser.add_argument("--expand", type=float, default=flow.expand_threshold)
    parser.add_argument("--cluster-radius", type=float, default=flow.cluster_radius)
    parser.add_argument("--min-points", type=int, default=flow.min_points)
    parser.add_argument("--min-tracks", type=int, default=flow.min_tracks)
    parser.add_argument("--box-pad", type=float, default=flow.box_pad)
    parser.add_argument("--flow-min-area", type=float, default=flow.min_area)
    parser.add_argument("--flow-max-area-frac", type=float, default=flow.max_area_frac)
    parser.add_argument("--flow-max-boxes", type=int, default=flow.max_boxes)
    return parser.parse_args()


def configs_from_args(args: argparse.Namespace):
    segment = dict(
        k=args.k,
        min_size=args.min_size,
        sigma=args.sigma,
        min_area=args.min_area,
        max_area_frac=args.max_area_frac,
        min_fill=args.min_fill,
        max_boxes=args.max_boxes,
        process_width=args.process_width,
    )
    felz = FelzenszwalbConfig(**segment)
    pav = PavementConfig(
        **segment,
        trap_top=args.trap_top,
        trap_top_width=args.trap_top_width,
        hist_bins=args.hist_bins,
        match_threshold=args.match_threshold,
        reject_fraction=args.reject_fraction,
        gray_saturation=args.gray_saturation,
    )
    flow = FlowConfig(
        max_corners=args.max_corners,
        quality=args.quality,
        min_distance=args.min_distance,
        lk_window=args.lk_window,
        expand_threshold=args.expand,
        cluster_radius=args.cluster_radius,
        min_points=args.min_points,
        min_tracks=args.min_tracks,
        box_pad=args.box_pad,
        min_area=args.flow_min_area,
        max_area_frac=args.flow_max_area_frac,
        max_boxes=args.flow_max_boxes,
    )
    track = TrackConfig(iou_match=args.iou_match, max_missed=args.max_missed)
    hit = CollisionConfig(
        frame_gap=args.frame_gap,
        confirm_hits=args.confirm_hits,
        min_growth_ratio=args.min_growth_ratio,
        max_ttc_seconds=args.max_ttc_seconds,
    )
    return felz, pav, flow, track, hit


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
    stream = next(item for item in info["streams"] if "width" in item)
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
        proc.wait(timeout=30)


def draw_lines(frame, lines: list[str]) -> None:
    y = 20
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        y += 18


def hud_lines(method: str, cfg, source_t: float, n_hit: int, n_boxes: int, frame_idx: int) -> list[str]:
    status = f"t={source_t:6.1f}s   frame {frame_idx}   boxes {n_boxes}   collisions {n_hit}"
    if method == "felzenszwalb":
        return [
            f"felzenszwalb   k={cfg.k:.0f}   min_size={cfg.min_size}   sigma={cfg.sigma:.1f}",
            status,
            (
                f"min area {cfg.min_area:.0f}px   fill {cfg.min_fill:.2f}   "
                f"max area {cfg.max_area_frac:.0%}   width {cfg.process_width}"
            ),
        ]
    if method == "pavement_mask":
        return [
            (
                f"pavement_mask   trap {cfg.trap_top:.2f}   bins {cfg.hist_bins}   "
                f"match {cfg.match_threshold:.2f}   reject {cfg.reject_fraction:.2f}"
            ),
            status,
            (
                f"k={cfg.k:.0f}  sigma={cfg.sigma:.1f}  min_size={cfg.min_size}   "
                f"min area {cfg.min_area:.0f}   max area {cfg.max_area_frac:.0%}"
            ),
        ]
    return [
        (
            f"sparse_flow   corners {cfg.max_corners}   expand {cfg.expand_threshold:.2f}px   "
            f"radius {cfg.cluster_radius:.0f}   min pts {cfg.min_points}"
        ),
        status,
        (
            f"quality {cfg.quality:.3f}   lk {cfg.lk_window}   min dist {cfg.min_distance:.0f}   "
            f"min area {cfg.min_area:.0f}"
        ),
    ]


def draw_trapezoid(frame, top: float, top_width: float) -> None:
    height, width = frame.shape[:2]
    polygon = trapezoid_polygon(width, height, top, top_width)
    cv2.polylines(frame, [polygon], True, (40, 180, 255), 1, cv2.LINE_AA)


def _largest_fraction(regions, frame_area: float) -> float:
    if not regions or frame_area <= 0:
        return 0.0
    largest = 0.0
    for region in regions:
        area = max(0.0, region.x2 - region.x1) * max(0.0, region.y2 - region.y1)
        if area > largest:
            largest = area
    return largest / frame_area


def process_clip(
    source: Path,
    output: Path,
    start: float,
    end: float,
    width: int,
    height: int,
    fps: float,
    method: str,
    cfg,
    detect,
    track_cfg: TrackConfig,
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

    tracker = RegionTracker(track_cfg.iou_match, track_cfg.max_missed)
    memory = TrackMemory(hit.frame_gap)
    latch = HitLatch(hit.confirm_hits)
    frame_idx = 0
    hit_flags = 0
    hit_tracks: set[int] = set()
    box_total = 0
    frames_with_boxes = 0
    max_box_frac = 0.0
    frame_area = float(width * height)

    try:
        for frame in iter_frames(source, start, end - start, width, height):
            frame_idx += 1
            regions = detect(frame)
            tracked = tracker.update(regions)
            box_total += len(regions)
            if regions:
                frames_with_boxes += 1
                max_box_frac = max(max_box_frac, _largest_fraction(regions, frame_area))

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
            if method == "pavement_mask":
                draw_trapezoid(vis, cfg.trap_top, cfg.trap_top_width)
            draw_margin_frame(vis, hit.frame_margin_horizontal, hit.frame_margin_vertical)
            draw_forecasts(vis, forecasts, current)
            hits = [forecast for forecast in forecasts if forecast.will_collide]
            source_t = start + frame_idx / fps
            draw_lines(vis, hud_lines(method, cfg, source_t, len(hits), len(current), frame_idx))
            writer.write(vis)

            for forecast in hits:
                hit_flags += 1
                hit_tracks.add(forecast.track_id)

            if frame_idx % 30 == 0:
                print(
                    f"    {frame_idx} frames, {len(current)} boxes, {len(hits)} hits",
                    flush=True,
                )
    finally:
        writer.release()

    if frame_idx == 0:
        output.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg returned no frames for {start:.1f}s–{end:.1f}s")

    return {
        "frames": frame_idx,
        "hit_flags": hit_flags,
        "hit_tracks": sorted(hit_tracks),
        "mean_boxes": box_total / frame_idx,
        "frames_with_boxes": frames_with_boxes,
        "max_box_frac": max_box_frac,
    }


def _write_index(out_dir: Path, header: dict, entry: dict) -> None:
    path = out_dir / "index.json"
    if path.exists():
        index = json.loads(path.read_text(encoding="utf-8"))
    else:
        index = {"methods": []}
    index.update(header)
    methods = [item for item in index.get("methods", []) if item.get("method") != entry["method"]]
    methods.append(entry)
    methods.sort(key=lambda item: METHODS.index(item["method"]))
    index["methods"] = methods
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def detector_for(method: str, felz: FelzenszwalbConfig, pav: PavementConfig, flow: FlowConfig):
    """A fresh detector. Flow keeps the previous frame, so each clip needs its own."""
    if method == "felzenszwalb":
        return lambda frame: find_felzenszwalb(frame, felz)
    if method == "pavement_mask":
        return lambda frame: find_pavement(frame, pav)
    return FlowBoxes(flow).boxes


def render_method(
    method: str,
    source: Path,
    out_dir: Path,
    width: int,
    height: int,
    fps: float,
    duration: float,
    cfg,
    felz: FelzenszwalbConfig,
    pav: PavementConfig,
    flow: FlowConfig,
    track: TrackConfig,
    hit: CollisionConfig,
) -> dict:
    folder = out_dir / method
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "about.txt").write_text(ABOUT[method] + "\n", encoding="utf-8")
    print(f"\n=== {method} ===", flush=True)
    print(ABOUT[method], flush=True)
    clips = []
    for filename, start, end in CLIPS:
        if start < 0 or end > duration + 0.05:
            raise SystemExit(f"{filename} is outside the source ({duration:.1f}s)")
        output = folder / filename
        print(f"rendering {method}/{filename}  {start:.1f}s – {end:.1f}s", flush=True)
        result = process_clip(
            source,
            output,
            start,
            end,
            width,
            height,
            fps,
            method,
            cfg,
            detector_for(method, felz, pav, flow),
            track,
            hit,
        )
        payload = {
            "method": method,
            "about": ABOUT[method],
            "source": str(source),
            "output": str(output),
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": result["frames"] / fps if fps else 0.0,
            "width": width,
            "height": height,
            "fps": fps,
            "frames": result["frames"],
            "hit_flags": result["hit_flags"],
            "hit_tracks": result["hit_tracks"],
            "mean_boxes": round(result["mean_boxes"], 3),
            "frames_with_boxes": result["frames_with_boxes"],
            "max_box_frac": round(result["max_box_frac"], 4),
            "segmentation_backend": segmentation_backend() if method != "sparse_flow" else None,
            "config": asdict(cfg),
            "track": asdict(track),
            "collision": asdict(hit),
        }
        output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"  {result['frames']} frames, {result['hit_flags']} hit flags, "
            f"mean boxes {result['mean_boxes']:.2f}, max box {result['max_box_frac']:.0%}",
            flush=True,
        )
        clips.append(
            {
                "file": filename,
                "start_seconds": start,
                "end_seconds": end,
                "frames": result["frames"],
                "duration_seconds": payload["duration_seconds"],
                "hit_flags": result["hit_flags"],
                "hit_tracks": result["hit_tracks"],
                "mean_boxes": payload["mean_boxes"],
                "frames_with_boxes": result["frames_with_boxes"],
                "max_box_frac": payload["max_box_frac"],
            }
        )
    return {
        "method": method,
        "about": ABOUT[method],
        "output_dir": str(folder),
        "segmentation_backend": segmentation_backend() if method != "sparse_flow" else None,
        "config": asdict(cfg),
        "clips": clips,
    }


def main() -> None:
    args = parse_args()
    felz, pav, flow, track, hit = configs_from_args(args)
    validate_felzenszwalb(felz)
    validate_pavement(pav)
    validate_flow(flow)
    validate_track(track)

    source = Path(args.source)
    out_dir = Path(args.output_dir)
    if not source.exists():
        raise SystemExit(f"missing video: {source}")

    width, height, fps, duration = probe_video(source)
    print(f"source {source}", flush=True)
    print(f"{width}x{height} @ {fps:.2f} fps, {duration:.1f}s", flush=True)
    print(f"segmentation backend: {segmentation_backend()}", flush=True)

    chosen = METHODS if args.only == "all" else (args.only,)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = {
        "source": str(source),
        "width": width,
        "height": height,
        "fps": fps,
        "segmentation_backend": segmentation_backend(),
    }
    configs = {"felzenszwalb": felz, "pavement_mask": pav, "sparse_flow": flow}
    for method in chosen:
        entry = render_method(
            method,
            source,
            out_dir,
            width,
            height,
            fps,
            duration,
            configs[method],
            felz,
            pav,
            flow,
            track,
            hit,
        )
        _write_index(out_dir, header, entry)
    print(f"\ndone. files are in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
