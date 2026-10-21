"""Locked 16-frame / 3-second sampling contract."""

from __future__ import annotations

import hashlib
import random
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


def group_anchor_seed(scenario_group_id: str, seed: int) -> int:
    """Stable 64-bit seed. Do not use Python's salted hash()."""
    blob = f"{int(seed)}\0{scenario_group_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "little")


def sample_group_anchors(
    scenario_group_id: str,
    *,
    seed: int,
    n_pre: int,
    n_post: int,
    t_min: int,
    t_max: int,
    fork_frame: int,
) -> list[int]:
    """Without-replacement pre/post draw, shared by every clip in the sample folder."""
    lo = int(t_min)
    hi = int(t_max)
    fork = int(fork_frame)
    if hi < lo:
        raise ValueError(f"empty anchor range [{lo}, {hi}]")
    pre = list(range(lo, min(fork, hi + 1)))
    post = list(range(max(fork, lo), hi + 1))
    n_pre = int(n_pre)
    n_post = int(n_post)
    if n_pre < 0 or n_post < 0 or n_pre + n_post < 1:
        raise ValueError(f"need a positive number of anchors, got n_pre={n_pre} n_post={n_post}")
    if n_pre > len(pre):
        raise ValueError(f"n_pre={n_pre} > pre-fork pool {len(pre)} ([{lo}, {fork}))")
    if n_post > len(post):
        raise ValueError(f"n_post={n_post} > post-fork pool {len(post)} ([{fork}, {hi}])")
    rng = random.Random(group_anchor_seed(scenario_group_id, seed))
    picked = rng.sample(pre, n_pre) + rng.sample(post, n_post)
    out = sorted(int(t) for t in picked)
    if len(set(out)) != len(out):
        raise RuntimeError(f"duplicate anchors for {scenario_group_id}: {out}")
    return out


def planned_anchors_per_clip(cfg) -> int:
    if str(cfg.get("sampling.anchor_mode", "stride")) == "group_random":
        return int(cfg.get("sampling.n_pre_fork_anchors", 26)) + int(
            cfg.get("sampling.n_post_fork_anchors", 14)
        )
    n_frames = int(cfg.get("data.expected_frames", 450))
    return len(
        iter_anchors(
            n_frames,
            int(cfg.get("sampling.train_anchor_stride", 15)),
            cfg.get("sampling.frame_offsets"),
            cfg.get("sampling.horizons_frames"),
            allow_partial_horizons=bool(cfg.get("sampling.allow_partial_horizons", True)),
        )
    )


def anchors_from_cfg(cfg, scenario_group_id: str, *, n_frames: int | None = None) -> list[int]:
    """Train/val/test anchors for one sample folder. Identical for all four clips."""
    n_frames = int(n_frames if n_frames is not None else cfg.get("data.expected_frames", 450))
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
    allow = bool(cfg.get("sampling.allow_partial_horizons", True))
    lo, hi = valid_anchor_range(n_frames, offsets, horizons, allow_partial_horizons=allow)
    mode = str(cfg.get("sampling.anchor_mode", "stride"))
    if mode == "group_random":
        t_min = max(lo, int(cfg.get("sampling.random_anchor_t_min", lo)))
        t_max = min(hi, int(cfg.get("sampling.random_anchor_t_max", hi)))
        return sample_group_anchors(
            scenario_group_id,
            seed=int(cfg.get("project.seed", 1337)),
            n_pre=int(cfg.get("sampling.n_pre_fork_anchors", 26)),
            n_post=int(cfg.get("sampling.n_post_fork_anchors", 14)),
            t_min=t_min,
            t_max=t_max,
            fork_frame=int(cfg.get("sampling.fork_frame", 365)),
        )
    if mode != "stride":
        raise ValueError(f"unsupported sampling.anchor_mode {mode}")
    return iter_anchors(
        n_frames,
        int(cfg.get("sampling.train_anchor_stride", 15)),
        offsets,
        horizons,
        allow_partial_horizons=allow,
    )
