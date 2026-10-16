"""Locked 16-frame / 3-second sampling contract."""

from __future__ import annotations

from typing import Sequence

FRAME_OFFSETS: tuple[int, ...] = (
    -90, -84, -78, -72, -66, -60, -54, -48,
    -42, -36, -30, -24, -18, -12, -6, 0,
)

HORIZONS_FRAMES: tuple[int, ...] = (30, 45, 60)


def input_indices(anchor: int, offsets: Sequence[int] | None = None) -> list[int]:
    offs = list(FRAME_OFFSETS if offsets is None else offsets)
    idx = [int(anchor) + int(o) for o in offs]
    if any(i > int(anchor) for i in idx):
        raise ValueError(f"future leakage: indices {idx} for anchor {anchor}")
    if idx[-1] != int(anchor):
        raise ValueError(f"last input frame must be the anchor, got {idx[-1]} vs {anchor}")
    return idx


def target_indices(anchor: int, horizons: Sequence[int] | None = None) -> dict[str, int]:
    hs = list(HORIZONS_FRAMES if horizons is None else horizons)
    if len(hs) != 3:
        raise ValueError("exactly three horizons required")
    t = int(anchor)
    return {
        "now": t,
        "1s": t + int(hs[0]),
        "1_5s": t + int(hs[1]),
        "2s": t + int(hs[2]),
    }


def available_horizons(
    anchor: int,
    n_frames: int,
    horizons: Sequence[int] | None = None,
) -> dict[str, bool]:
    """True when that target frame exists. Missing later horizons are not an error."""
    n = int(n_frames)
    return {name: 0 <= idx < n for name, idx in target_indices(anchor, horizons).items()}


def valid_anchor_range(
    n_frames: int,
    offsets: Sequence[int] | None = None,
    horizons: Sequence[int] | None = None,
    *,
    allow_partial_horizons: bool = True,
) -> tuple[int, int]:
    offs = list(FRAME_OFFSETS if offsets is None else offsets)
    hs = list(HORIZONS_FRAMES if horizons is None else horizons)
    t_min = -min(offs)
    if allow_partial_horizons:
        # Need history + a real 1s frame: t+30 <= n_frames-1.
        # 450-frame clip → t_max=419; stride 15 last anchor is 405.
        # 1.5s/2s past the end are masked.
        t_max = int(n_frames) - 1 - min(hs)
    else:
        t_max = int(n_frames) - 1 - max(hs)
    return t_min, t_max


def iter_anchors(
    n_frames: int,
    stride: int,
    offsets=None,
    horizons=None,
    *,
    allow_partial_horizons: bool = True,
) -> list[int]:
    t_min, t_max = valid_anchor_range(
        n_frames,
        offsets,
        horizons,
        allow_partial_horizons=allow_partial_horizons,
    )
    if t_max < t_min:
        return []
    return list(range(int(t_min), int(t_max) + 1, int(stride)))
