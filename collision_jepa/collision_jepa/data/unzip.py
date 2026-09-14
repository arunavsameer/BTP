"""Extract the dataset zip.

The archive layout is::

    dataset_100/
      episode_XXXX_<scenario>/
        episode.json
        preview.mp4
        spatial_overlay.mp4                 (visualization only)
        spatial_annotations/spatial_annotations.json

We can optionally skip the (large) ``spatial_overlay.mp4`` files, which are not
used for training.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract_dataset(zip_path: str | Path, raw_dir: str | Path, skip_overlays: bool = True) -> Path:
    """Extract ``zip_path`` into ``raw_dir``.

    Returns the path to the top-level ``dataset_100`` directory.
    Files that already exist with the right size are skipped so re-running is cheap.
    """
    zip_path = Path(zip_path)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        n_written = 0
        n_skipped = 0
        for info in members:
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

    # Locate the dataset root (usually 'dataset_100').
    candidates = [p for p in raw_dir.iterdir() if p.is_dir() and p.name.startswith("dataset")]
    root = candidates[0] if candidates else raw_dir
    return root


def find_episode_dirs(dataset_root: str | Path) -> list[Path]:
    """Return sorted episode directories under the dataset root."""
    dataset_root = Path(dataset_root)
    eps = [p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("episode_")]
    return sorted(eps, key=lambda p: p.name)
