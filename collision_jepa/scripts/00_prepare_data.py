"""Stage 0 of the pipeline: unzip, split, and cache frames + heatmaps.

Run from the repo root:  python scripts/00_prepare_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when run as a plain script.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.config import Config  # noqa: E402
from collision_jepa.data.splits import make_split, read_episode_meta, save_split  # noqa: E402
from collision_jepa.data.unzip import extract_dataset, find_episode_dirs  # noqa: E402
from collision_jepa.data.video import cache_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Collision-JEPA dataset cache.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="Only process N episodes (debug).")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    zip_path = cfg.get("data.zip_path")
    raw_dir = cfg.get("data.raw_dir")
    cache_dir = Path(cfg.get("data.cache_dir"))
    skip_overlays = bool(cfg.get("data.skip_overlays", True))
    student_size = int(cfg.get("student.img_size"))
    cache_all = bool(cfg.get("data.cache_all_frames", True))

    print(f"[prepare] extracting {zip_path} -> {raw_dir} (skip_overlays={skip_overlays})")
    dataset_root = extract_dataset(zip_path, raw_dir, skip_overlays=skip_overlays)
    episode_dirs = find_episode_dirs(dataset_root)
    if args.limit:
        episode_dirs = episode_dirs[: args.limit]
    print(f"[prepare] found {len(episode_dirs)} episodes under {dataset_root}")

    # Which frames to cache (all, or only those the student/teacher touch).
    frame_indices = None
    if not cache_all:
        offsets = cfg.get("student.frame_offsets")
        tau = int(cfg.get("horizon.tau_frames"))
        needed: set[int] = set()
        n_frames = int(cfg.get("data.n_frames", 150))
        for t in range(-min(offsets), n_frames - tau):
            for off in offsets:
                needed.add(t + off)
            needed.add(t + tau)
        frame_indices = sorted(needed)

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, ep in enumerate(episode_dirs):
        info = cache_episode(ep, cache_dir, student_size, frame_indices=frame_indices)
        manifest.append(info)
        if (i + 1) % 10 == 0 or i == len(episode_dirs) - 1:
            print(f"[prepare] cached {i + 1}/{len(episode_dirs)} episodes")

    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Episode-level stratified split.
    split = make_split(
        episode_dirs,
        val_fraction=float(cfg.get("data.val_fraction", 0.2)),
        seed=int(cfg.get("seed", 1337)),
    )
    save_split(split, cfg.get("data.split_file"))
    fam_counts: dict[str, int] = {}
    for name in split["train"] + split["val"]:
        fam_counts[split["families"][name]] = fam_counts.get(split["families"][name], 0) + 1
    n_rare = sum(1 for v in split["has_rare"].values() if v)
    print(f"[prepare] split: {len(split['train'])} train / {len(split['val'])} val")
    print(f"[prepare] families: {fam_counts}")
    print(f"[prepare] episodes containing NEAR_MISS/CRITICAL: {n_rare}")
    print(f"[prepare] split written to {cfg.get('data.split_file')}")


if __name__ == "__main__":
    main()
