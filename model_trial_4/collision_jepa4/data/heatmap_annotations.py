"""Load and validate 5×5 spatial heatmaps by frame ID."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class HeatmapError(ValueError):
    """Corrupt or incomplete spatial annotations."""


def _as_frame_id(raw: object) -> int:
    if isinstance(raw, int):
        return int(raw)
    text = str(raw).strip()
    if text.startswith("frame_"):
        text = text.split("_", 1)[1]
    return int(text)


def load_heatmaps(
    path: str | Path,
    *,
    expected_frames: int | None = 450,
    grid: int = 5,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    """Return float32 heatmaps indexed by integer frame ID.

    Missing IDs are not filled with zeros. The returned array is dense only when
    IDs form a contiguous 0..N-1 range covering ``expected_frames``.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "frames" in data:
        raw_frames = data["frames"]
        k = data.get("k")
        if k is not None and int(k) != int(grid):
            raise HeatmapError(f"{path}: k={k} does not match target grid {grid}")
    elif isinstance(data, list):
        raw_frames = data
    else:
        raise HeatmapError(f"{path}: unrecognized heatmap JSON schema")

    by_id: dict[int, np.ndarray] = {}
    duplicates: list[int] = []
    for item in raw_frames:
        if not isinstance(item, dict) or "matrix" not in item:
            raise HeatmapError(f"{path}: frame entry missing matrix")
        fid = _as_frame_id(item.get("frame_id", item.get("id")))
        mat = np.asarray(item["matrix"], dtype=np.float32)
        if mat.shape != (grid, grid):
            raise HeatmapError(f"{path}: frame {fid} has shape {mat.shape}, expected ({grid},{grid})")
        if not np.isfinite(mat).all():
            raise HeatmapError(f"{path}: frame {fid} contains non-finite values")
        if float(mat.min()) < value_min - 1e-6 or float(mat.max()) > value_max + 1e-6:
            raise HeatmapError(
                f"{path}: frame {fid} values outside [{value_min},{value_max}] "
                f"(min={mat.min()}, max={mat.max()})"
            )
        mat = np.clip(mat, value_min, value_max)
        if fid in by_id:
            duplicates.append(fid)
        by_id[fid] = mat

    if duplicates:
        raise HeatmapError(f"{path}: duplicate frame IDs {sorted(set(duplicates))[:8]}")
    if not by_id:
        raise HeatmapError(f"{path}: no heatmap frames")

    ids = sorted(by_id)
    if expected_frames is not None:
        expected_ids = list(range(int(expected_frames)))
        missing = [i for i in expected_ids if i not in by_id]
        extra = [i for i in ids if i not in expected_ids]
        if missing or extra:
            raise HeatmapError(
                f"{path}: frame-ID coverage error missing={missing[:8]} extra={extra[:8]} "
                f"n={len(ids)} expected={expected_frames}"
            )
        stacked = np.stack([by_id[i] for i in expected_ids], axis=0)
    else:
        if ids != list(range(ids[0], ids[-1] + 1)):
            missing = [i for i in range(ids[0], ids[-1] + 1) if i not in by_id]
            raise HeatmapError(f"{path}: non-contiguous frame IDs, missing={missing[:8]}")
        stacked = np.stack([by_id[i] for i in ids], axis=0)
    return stacked.astype(np.float32, copy=False)


def targets_for_anchor(
    heatmaps: np.ndarray,
    anchor: int,
    horizons: Sequence[int],
    *,
    allow_missing: bool = False,
) -> dict[str, np.ndarray]:
    t = int(anchor)
    hs = list(horizons)
    n = heatmaps.shape[0]
    needed = {
        "h_now": t,
        "h_1s": t + hs[0],
        "h_1_5s": t + hs[1],
        "h_2s": t + hs[2],
    }
    missing = [i for i in needed.values() if i < 0 or i >= n]
    if missing and not allow_missing:
        raise HeatmapError(f"anchor {t} targets {list(needed.values())} out of range n={n}")
    zeros = np.zeros(heatmaps.shape[1:], dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for key, i in needed.items():
        if i < 0 or i >= n:
            out[key] = zeros.copy()
        else:
            out[key] = np.asarray(heatmaps[i], dtype=np.float32)
    return out
