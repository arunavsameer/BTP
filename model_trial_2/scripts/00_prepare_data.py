"""Build catalog, split, and cache 160px (student) + 256px (teacher) frames."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.config import Config  # noqa: E402
from cvjepa.data import build_catalog, load_json, make_split, save_json  # noqa: E402
from cvjepa.data.video import cache_episode, load_heatmaps  # noqa: E402


def _cache_one(args: tuple) -> dict:
    path, cache_dir, sizes, grid, key, mode = args
    return cache_episode(path, cache_dir, sizes, target_grid=grid, cache_key=key, upsample_mode=mode)


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
    parser.add_argument("--reset-split", action="store_true")
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
    sizes = sorted({int(cfg.get("student.img_size")), int(cfg.get("teacher.img_size"))})
    grid = int(cfg.get("data.target_grid", 5))

    catalog = build_catalog(cfg.get("data.sources"), seed=int(cfg.get("seed", 1337)))
    if args.limit:
        catalog = catalog[: args.limit]
    save_json(catalog, cfg.get("data.catalog_file"))
    by_src: dict[str, int] = {}
    for row in catalog:
        by_src[row["source"]] = by_src.get(row["source"], 0) + 1
    print(f"[prepare] catalog={len(catalog)}  by_source={by_src}  sizes={sizes}  grid={grid}")
    seen: set[str] = set()
    for row in catalog:
        if row["source"] in seen:
            continue
        seen.add(row["source"])
        native = load_heatmaps(Path(row["path"]) / "spatial_annotations" / "spatial_annotations.json")
        print(f"[prepare] native {row['source']}: {native.shape[-2]}x{native.shape[-1]}")

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
    print(
        f"[prepare] split: {len(split['train'])} train / {len(split['val'])} val / "
        f"{len(split.get('test') or [])} test"
    )
    jobs = [(row["path"], str(cache_dir), sizes, grid, row["key"], mode) for row in catalog]
    _run_jobs(jobs, args.workers)
    print(f"[prepare] done. cache={cache_dir}")


if __name__ == "__main__":
    main()
