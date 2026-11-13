"""Render each noise-reduction idea on the same three scenes.

The clips are the ones already written to impl2/output/ (seed 0), so a
folder can be compared frame for frame with the others. One idea per folder.
"""

import json
from dataclasses import replace
from pathlib import Path

from config import Config, validate
from run import DEFAULT_SOURCE, CollisionConfig, probe_video, process_clip

# Same windows as output/clips.json from the first run.
CLIPS = [
    ("clip_01_753s-770s.mp4", 752.5494252392793, 769.6522831434336),
    ("clip_02_1177s-1194s.mp4", 1176.7697669163908, 1194.3261405232338),
    ("clip_03_2201s-2221s.mp4", 2201.411274984765, 2220.63338424239),
]

OUT = Path(__file__).resolve().parent / "output" / "strategies"


def strategies() -> list[tuple[str, str, Config, CollisionConfig]]:
    base = Config()
    hit = CollisionConfig()
    calmer_hit = replace(
        hit,
        frame_gap=20,
        confirm_hits=6,
        min_growth_ratio=1.12,
        frame_margin_vertical=1.5,
    )
    combined_hit = replace(
        hit,
        frame_gap=15,
        confirm_hits=5,
        min_growth_ratio=1.08,
        frame_margin_vertical=1.5,
    )
    return [
        (
            "00_baseline",
            "Current per-channel color test, current filters, full forecast drawing.",
            base,
            hit,
        ),
        (
            "01_euclidean",
            "Same knobs, but closeness is Euclidean distance in RGB instead of a per-channel gap.",
            replace(base, color_metric="euclidean"),
            hit,
        ),
        (
            "02_stricter_sizes",
            "Tighter tolerance, larger minimum area, smaller maximum, higher fill, only 8 boxes.",
            replace(base, color_tolerance=0.10, min_area=2000, max_area_frac=0.10, min_fill=0.60, max_boxes=8),
            hit,
        ),
        (
            "03_shape_border",
            "Drop long thin boxes (aspect over 3.5) and any box that touches the frame edge.",
            replace(base, max_aspect=3.5, drop_border=True),
            hit,
        ),
        (
            "04_merge_colors",
            "Join nearby boxes whose seed colors are within 50 levels (Euclidean).",
            replace(base, merge_gap=40, merge_color=50),
            hit,
        ),
        (
            "05_drop_gray",
            "Drop low-saturation seeds. Gray sidewalk and sky go; black clothes go with them.",
            replace(base, min_saturation=0.28),
            hit,
        ),
        (
            "06_stable_tracks",
            "Hide a box until it has matched for 8 frames, and average its box over 5 frames.",
            replace(base, min_track_frames=8, smooth_frames=5),
            hit,
        ),
        (
            "07_calmer_hits",
            "Both sides of the box must grow, jitter is rejected, and the hit zone is 1.5x tall instead of 10x.",
            replace(base, require_both_axes=True, max_size_jitter=0.15),
            calmer_hit,
        ),
        (
            "08_scene_growth",
            "A box must grow at least 10% faster than the median box in that frame.",
            replace(base, use_scene_growth=True, scene_growth_margin=1.10),
            hit,
        ),
        (
            "09_quiet_draw",
            "Ordinary blobs are a plain rectangle. Paths and HIT text are drawn only for a collision.",
            replace(base, quiet_draw=True),
            hit,
        ),
        (
            "10_combined",
            "Wider color match and a lower area and fill floor, so a shirt, a bare leg, or jeans can stay one box. "
            "Thin boxes and boxes on the frame edge are still dropped. A box is drawn from its first frame. "
            "Hits stay calmer, and ordinary blobs are drawn quietly.",
            replace(
                base,
                color_tolerance=0.18,
                min_area=500,
                max_area_frac=0.15,
                min_fill=0.40,
                max_boxes=10,
                max_aspect=3.5,
                drop_border=True,
                min_track_frames=1,
                smooth_frames=4,
                require_both_axes=True,
                max_size_jitter=0.18,
                quiet_draw=True,
            ),
            combined_hit,
        ),
    ]


def main() -> None:
    source = DEFAULT_SOURCE
    if not source.exists():
        raise SystemExit(f"missing video: {source}")
    width, height, fps, _duration = probe_video(source)
    OUT.mkdir(parents=True, exist_ok=True)
    index = []
    for name, note, cfg, hit in strategies():
        validate(cfg)
        folder = OUT / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "about.txt").write_text(note + "\n", encoding="utf-8")
        print(f"\n=== {name} ===")
        print(note)
        clips = []
        for filename, start, end in CLIPS:
            output = folder / filename
            print(f"rendering {name}/{filename}")
            result = process_clip(source, output, start, end, width, height, fps, cfg, hit, label=name)
            payload = {
                "strategy": name,
                "note": note,
                "source": str(source),
                "output": str(output),
                "start_seconds": start,
                "end_seconds": end,
                "frames": result["frames"],
                "config": cfg.__dict__,
                "collision": hit.__dict__,
                "hit_flags": len(result["collisions"]),
            }
            output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  {result['frames']} frames, {len(result['collisions'])} hit flags")
            clips.append({"file": filename, "hits": len(result["collisions"]), "frames": result["frames"]})
        index.append({"name": name, "note": note, "clips": clips})
        (OUT / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\ndone. compare the folders in {OUT}")


if __name__ == "__main__":
    main()
