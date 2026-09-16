"""Stage 0 of the pipeline: unzip a balanced subset, split, and cache frames + heatmaps.

Takes mostly native k=5 episodes (``data.k5_fraction``) plus a smaller equal draw
from the other folders, until ``data.target_episodes`` (~500).

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
from collision_jepa.data.splits import make_count_split, save_split  # noqa: E402
from collision_jepa.data.unzip import (  # noqa: E402
    episode_cache_name,
    extract_selected_episodes,
    list_zip_subsets,
    resolve_dataset_root,
    sample_equal_from_subsets,
    sample_with_k5_bias,
)
from collision_jepa.data.video import cache_episode  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Collision-JEPA dataset cache.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="Only process N episodes (debug).")
    parser.add_argument("--total", type=int, default=0, help="Override data.target_episodes.")
    args = parser.parse_args()

    cfg = Config.load(args.config)

    zip_path = cfg.get("data.zip_path")
    raw_dir = cfg.get("data.raw_dir")
    cache_dir = Path(cfg.get("data.cache_dir"))
    skip_overlays = bool(cfg.get("data.skip_overlays", True))
    student_size = int(cfg.get("student.img_size"))
    cache_all = bool(cfg.get("data.cache_all_frames", True))
    grid = int(cfg.get("data.grid", 5))
    total = int(args.total or cfg.get("data.target_episodes", 500))
    seed = int(cfg.get("seed", 1337))

    print(f"[prepare] listing subsets in {zip_path}")
    by_subset = list_zip_subsets(zip_path)
    if not by_subset:
        raise SystemExit(f"[prepare] no episode folders found in {zip_path}")
    for name, eps in by_subset.items():
        print(f"[prepare]   {name}: {len(eps)} episodes")

    k5_subsets = cfg.get("data.k5_subsets") or []
    k5_fraction = float(cfg.get("data.k5_fraction", 0.8))
    if k5_subsets:
        selected = sample_with_k5_bias(
            by_subset, total=total, seed=seed, k5_subsets=list(k5_subsets), k5_fraction=k5_fraction
        )
        print(f"[prepare] sampled with k=5 bias (fraction={k5_fraction}) from {k5_subsets}")
    else:
        selected = sample_equal_from_subsets(by_subset, total=total, seed=seed)
        print("[prepare] sampled equally from each folder")
    n_sel = sum(len(v) for v in selected.values())
    print(f"[prepare] sampled {n_sel} episodes (target={total}):")
    for name, eps in selected.items():
        print(f"[prepare]   {name}: {len(eps)}")

    print(f"[prepare] extracting selected episodes -> {raw_dir} (skip_overlays={skip_overlays})")
    dataset_root = extract_selected_episodes(zip_path, raw_dir, selected, skip_overlays=skip_overlays)
    dataset_root = resolve_dataset_root(raw_dir)

    episode_dirs: list[Path] = []
    cache_names: list[str] = []
    missing = 0
    for subset, eps in selected.items():
        for ep in eps:
            d = dataset_root / subset / ep
            if not d.is_dir():
                missing += 1
                print(f"[prepare] missing after extract: {subset}/{ep}")
                continue
            episode_dirs.append(d)
            cache_names.append(episode_cache_name(d, dataset_root))
    if missing:
        print(f"[prepare] warning: {missing} selected episodes were not on disk")
    if args.limit:
        episode_dirs = episode_dirs[: args.limit]
        cache_names = cache_names[: args.limit]
    print(f"[prepare] {len(episode_dirs)} episodes under {dataset_root}")

    sample_path = cache_dir / "sample.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(
            {
                "zip_path": str(zip_path),
                "target": total,
                "seed": seed,
                "counts": {k: len(v) for k, v in selected.items()},
                "selected": selected,
                "cache_names": cache_names,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

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

    manifest = []
    for i, (ep, name) in enumerate(zip(episode_dirs, cache_names)):
        info = cache_episode(
            ep,
            cache_dir,
            student_size,
            frame_indices=frame_indices,
            cache_name=name,
            grid=grid,
        )
        manifest.append(info)
        if (i + 1) % 10 == 0 or i == len(episode_dirs) - 1:
            print(f"[prepare] cached {i + 1}/{len(episode_dirs)} episodes")

    (cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Episode-level stratified split (ids = unique cache names).
    split = make_count_split(
        episode_dirs,
        n_val=int(cfg.get("data.val_count", 50)),
        n_test=int(cfg.get("data.test_count", 50)),
        seed=seed,
        names=cache_names,
    )
    save_split(split, cfg.get("data.split_file"))
    fam_counts: dict[str, int] = {}
    for name in split["train"] + split["val"] + split["test"]:
        fam_counts[split["families"][name]] = fam_counts.get(split["families"][name], 0) + 1
    n_rare = sum(1 for v in split["has_rare"].values() if v)
    print(
        f"[prepare] split: {len(split['train'])} train / {len(split['val'])} val / {len(split['test'])} test"
    )
    print(f"[prepare] families: {fam_counts}")
    print(f"[prepare] episodes containing NEAR_MISS/CRITICAL: {n_rare}")
    print(f"[prepare] split written to {cfg.get('data.split_file')}")


if __name__ == "__main__":
    main()
