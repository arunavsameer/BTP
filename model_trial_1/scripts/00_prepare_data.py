"""Build the catalog, split, and cache 160px frames + k×k heatmaps."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.config import Config  # noqa: E402
from rs_jepa.data import build_catalog, load_json, make_split, save_json  # noqa: E402
from rs_jepa.data.video import cache_episode  # noqa: E402


def _cache_one(args: tuple) -> dict:
    path, cache_dir, size, grid, key, mode = args
    return cache_episode(path, cache_dir, size, target_grid=grid, cache_key=key, upsample_mode=mode)


def _run_jobs(jobs: list[tuple], workers: int) -> list[dict]:
    manifest: list[dict] = []
    if workers <= 1:
        for i, job in enumerate(jobs):
            manifest.append(_cache_one(job))
            if (i + 1) % 10 == 0 or i + 1 == len(jobs):
                print(f"[prepare] cached {i + 1}/{len(jobs)}", flush=True)
        return manifest
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_cache_one, job) for job in jobs]
        for fut in as_completed(futs):
            manifest.append(fut.result())
            done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f"[prepare] cached {done}/{len(jobs)}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--heatmaps-only", action="store_true", help="Rewrite heatmaps.npy; keep split.json.")
    parser.add_argument("--reset-split", action="store_true", help="Rebuild train/val/test even if split.json exists.")
    args = parser.parse_args()

    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    cfg = Config.load(args.config)
    cache_dir = Path(cfg.get("data.cache_dir"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    mode = str(cfg.get("data.heatmap_upsample", "nearest"))
    size = int(cfg.get("student.img_size"))
    grid = int(cfg.get("data.target_grid", 5))

    if args.heatmaps_only:
        catalog = load_json(cfg.get("data.catalog_file"))
        if args.limit:
            catalog = catalog[: args.limit]
        print(f"[prepare] heatmaps-only catalog={len(catalog)} upsample={mode}")
        jobs = [(row["path"], str(cache_dir), size, grid, row["key"], mode) for row in catalog]
        manifest = _run_jobs(jobs, args.workers)
        save_json(manifest, cache_dir / "manifest.json")
        print(f"[prepare] heatmaps rewritten. cache={cache_dir}")
        return

    catalog = build_catalog(cfg.get("data.sources"), seed=int(cfg.get("seed", 1337)))
    if args.limit:
        catalog = catalog[: args.limit]
    save_json(catalog, cfg.get("data.catalog_file"))
    by_src: dict[str, int] = {}
    for row in catalog:
        by_src[row["source"]] = by_src.get(row["source"], 0) + 1
    print(f"[prepare] catalog={len(catalog)}  by_source={by_src}  target_grid={grid}  upsample={mode}")

    split_path = Path(cfg.get("data.split_file"))
    if split_path.exists() and not args.reset_split:
        split = load_json(split_path)
        print(f"[prepare] keeping existing split {split_path}")
    else:
        split = make_split(
            catalog,
            val_fraction=float(cfg.get("data.val_fraction", 0.2)),
            seed=int(cfg.get("seed", 1337)),
            test_fraction=float(cfg.get("data.test_fraction", 0.0)),
        )
        save_json(split, split_path)
    fam_counts: dict[str, int] = {}
    hold = split["train"] + split["val"] + list(split.get("test") or [])
    for name in hold:
        fam_counts[split["families"][name]] = fam_counts.get(split["families"][name], 0) + 1
    n_rare = sum(1 for v in split["has_rare"].values() if v)
    print(
        f"[prepare] split: {len(split['train'])} train / {len(split['val'])} val / "
        f"{len(split.get('test') or [])} test"
    )
    print(f"[prepare] families: {fam_counts}")
    print(f"[prepare] NEAR_MISS/CRITICAL episodes: {n_rare}")

    jobs = [(row["path"], str(cache_dir), size, grid, row["key"], mode) for row in catalog]
    manifest = _run_jobs(jobs, args.workers)
    save_json(manifest, cache_dir / "manifest.json")
    print(f"[prepare] done. cache={cache_dir}")


if __name__ == "__main__":
    main()
