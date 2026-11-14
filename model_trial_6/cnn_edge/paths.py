"""Read-only caches; trial-6 write root."""

from __future__ import annotations

import os
from pathlib import Path

TRIAL6 = Path(__file__).resolve().parent.parent
TRIAL5 = Path("/storage/BTP/model_trial_5")
TRIAL3 = Path("/storage/BTP/model_trial_3")  # full collision_jepa3 sources (trial-5 tree is partial)
EXPECTED_DECODER_HASH = "77f78c8714667617"
OFFSETS_16 = [-90, -84, -78, -72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]
RUN_NAME = "cnn_edge_v1"


def rgb_cache() -> Path:
    env = os.environ.get("STUDENT_RGB_CACHE")
    if env:
        return Path(env)
    return TRIAL5 / "shared_rgb_cache_256_v1"


def parent_cache() -> Path:
    env = os.environ.get("STUDENT_PARENT_CACHE")
    if env:
        return Path(env)
    return TRIAL3 / "checkpoints" / "parent_target_cache_v2"


def parent_ckpt() -> Path:
    env = os.environ.get("STUDENT_PARENT_CKPT")
    if env:
        return Path(env)
    flat = TRIAL3 / "checkpoints" / "pair_cf_treatment_s1337_best.pt"
    if flat.is_file():
        return flat
    return TRIAL5 / "checkpoints" / "pair_cf_treatment_s1337_best.pt"


def stats_path() -> Path:
    env = os.environ.get("STUDENT_FDEC_STATS")
    if env:
        return Path(env)
    t5 = TRIAL5 / "student_runs" / "fdec_channel_stats.npz"
    if t5.is_file():
        return t5
    return TRIAL6 / "runs" / "fdec_channel_stats.npz"


def run_dir(name: str = RUN_NAME) -> Path:
    env = os.environ.get("STUDENT_RUN_DIR")
    if env:
        return Path(env)
    return TRIAL6 / "runs" / name
