"""Extract and index the dataset zip.

Expected archive layout::

    datasets/
      <subset_a>/
        episode_XXXX_<scenario>/
          episode.json
          preview.mp4
          spatial_overlay.mp4                 (visualization only)
          spatial_annotations/spatial_annotations.json
      <subset_b>/
        ...

Older single-folder zips (``dataset_100/episode_*``) still work.
We can skip ``spatial_overlay.mp4`` (not used for training) and extract only a
balanced subset of episodes.
"""

from __future__ import annotations

import random
import zipfile
from collections import defaultdict
from pathlib import Path


def list_zip_subsets(zip_path: str | Path) -> dict[str, list[str]]:
    """Return ``{subset_name: [episode_dir_name, ...]}`` from the zip index."""
    zip_path = Path(zip_path)
    by_subset: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            parts = name.replace("\\", "/").split("/")
            # datasets/<subset>/episode_*  or  <subset>/episode_* at zip root
            if len(parts) >= 3 and parts[0] == "datasets" and parts[2].startswith("episode_"):
                subset, ep = parts[1], parts[2]
            elif len(parts) >= 2 and parts[1].startswith("episode_") and parts[0] not in ("", "datasets"):
                subset, ep = parts[0], parts[1]
            else:
                continue
            if not subset or subset.startswith("."):
                continue
            by_subset[subset].add(ep)
    return {k: sorted(v) for k, v in sorted(by_subset.items())}


def sample_equal_from_subsets(
    by_subset: dict[str, list[str]],
    total: int = 500,
    seed: int = 1337,
) -> dict[str, list[str]]:
    """Take as evenly as possible from each subset until ``total`` episodes.

    If a subset is smaller than a fair share, take all of it and keep drawing
    round-robin from the others (e.g. 50 + 225 + 225 = 500).
    """
    rng = random.Random(seed)
    keys = sorted(by_subset)
    remaining = {k: list(by_subset[k]) for k in keys}
    for k in keys:
        rng.shuffle(remaining[k])
    chosen: dict[str, list[str]] = {k: [] for k in keys}
    n = 0
    while n < total:
        progressed = False
        for k in keys:
            if n >= total:
                break
            if remaining[k]:
                chosen[k].append(remaining[k].pop())
                n += 1
                progressed = True
        if not progressed:
            break
    for k in keys:
        chosen[k].sort()
    return chosen


def sample_with_k5_bias(
    by_subset: dict[str, list[str]],
    total: int = 500,
    seed: int = 1337,
    k5_subsets: list[str] | None = None,
    k5_fraction: float = 0.8,
) -> dict[str, list[str]]:
    """Draw ``k5_fraction`` of episodes from native k=5 packs, the rest equally from others."""
    k5_subsets = [k for k in (k5_subsets or []) if k in by_subset]
    if not k5_subsets:
        return sample_equal_from_subsets(by_subset, total=total, seed=seed)
    others = {k: v for k, v in by_subset.items() if k not in k5_subsets}
    n_k5 = int(round(total * k5_fraction))
    n_k5 = min(n_k5, sum(len(by_subset[k]) for k in k5_subsets))
    n_other = max(0, total - n_k5)
    k5_pool = {k: by_subset[k] for k in k5_subsets}
    chosen_k5 = sample_equal_from_subsets(k5_pool, total=n_k5, seed=seed)
    chosen_other = (
        sample_equal_from_subsets(others, total=n_other, seed=seed + 1) if others and n_other else {k: [] for k in others}
    )
    out: dict[str, list[str]] = {k: [] for k in by_subset}
    for k, eps in chosen_k5.items():
        out[k] = eps
    for k, eps in chosen_other.items():
        out[k] = eps
    return out


def extract_selected_episodes(
    zip_path: str | Path,
    raw_dir: str | Path,
    selected: dict[str, list[str]],
    skip_overlays: bool = True,
) -> Path:
    """Extract only the chosen episode folders into ``raw_dir``.

    Returns the ``datasets/`` root under ``raw_dir``.
    """
    zip_path = Path(zip_path)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    allowed = {(subset, ep) for subset, eps in selected.items() for ep in eps}

    n_written = 0
    n_skipped = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if skip_overlays and name.endswith("spatial_overlay.mp4"):
                continue
            parts = [p for p in name.split("/") if p]
            key = None
            if len(parts) >= 3 and parts[0] == "datasets":
                key = (parts[1], parts[2])
            elif len(parts) >= 2:
                key = (parts[0], parts[1])
            if key not in allowed:
                continue
            out_path = raw_dir / name
            if info.is_dir() or name.endswith("/"):
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            if out_path.exists() and out_path.stat().st_size == info.file_size:
                n_skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            n_written += 1

    print(f"[unzip] wrote {n_written} files, skipped {n_skipped} up-to-date files")
    root = raw_dir / "datasets"
    return root if root.exists() else raw_dir


def extract_dataset(zip_path: str | Path, raw_dir: str | Path, skip_overlays: bool = True) -> Path:
    """Extract ``zip_path`` into ``raw_dir`` (full archive, overlays optional)."""
    zip_path = Path(zip_path)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        n_written = 0
        n_skipped = 0
        for info in zf.infolist():
            name = info.filename
            if skip_overlays and name.endswith("spatial_overlay.mp4"):
                continue
            out_path = raw_dir / name
            if info.is_dir():
                out_path.mkdir(parents=True, exist_ok=True)
                continue
            if out_path.exists() and out_path.stat().st_size == info.file_size:
                n_skipped += 1
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            n_written += 1

    print(f"[unzip] wrote {n_written} files, skipped {n_skipped} up-to-date files")
    datasets = raw_dir / "datasets"
    if datasets.is_dir():
        return datasets
    candidates = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("dataset")]
    return candidates[0] if candidates else raw_dir


def find_episode_dirs(dataset_root: str | Path) -> list[Path]:
    """Return episode directories (recursive) that have preview + spatial labels."""
    dataset_root = Path(dataset_root)
    found: list[Path] = []
    for meta in dataset_root.rglob("episode.json"):
        d = meta.parent
        if (d / "preview.mp4").exists() and (d / "spatial_annotations" / "spatial_annotations.json").exists():
            found.append(d)
    return sorted(found, key=lambda p: str(p).replace("\\", "/"))


def episode_cache_name(episode_dir: str | Path, dataset_root: str | Path) -> str:
    """Stable unique cache id: ``<subset>__<episode_dir>`` (or just the folder name)."""
    episode_dir = Path(episode_dir)
    dataset_root = Path(dataset_root)
    try:
        rel = episode_dir.resolve().relative_to(dataset_root.resolve())
        parts = rel.parts
        if len(parts) >= 2:
            return "__".join(parts)
        return parts[-1]
    except ValueError:
        return episode_dir.name


def resolve_dataset_root(raw_dir: str | Path) -> Path:
    """Prefer ``raw_dir/datasets`` so leftover ``dataset_100`` is not mixed in."""
    raw_dir = Path(raw_dir)
    nested = raw_dir / "datasets"
    if nested.is_dir():
        return nested
    candidates = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("dataset")]
    return candidates[0] if candidates else raw_dir


def episode_dir_map(dataset_root: str | Path) -> dict[str, Path]:
    """Map cache episode names to on-disk directories."""
    dataset_root = Path(dataset_root)
    return {episode_cache_name(p, dataset_root): p for p in find_episode_dirs(dataset_root)}
