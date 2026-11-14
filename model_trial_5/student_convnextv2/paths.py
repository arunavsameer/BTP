"""Read-only Trial-3 parent artifacts; Trial-5 write roots. Local paths only."""

from __future__ import annotations

import os
from pathlib import Path

TRIAL5 = Path(__file__).resolve().parent.parent
TRIAL3 = Path("/storage/BTP/model_trial_3")
TEMP3 = Path("/storage/BTP/model_trial_3_temp")
WORKSPACE = Path("/storage/BTP")

EXPECTED_DECODER_HASH = "77f78c8714667617"
PARENT_CKPT_NAME = "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt"
PARENT_CKPT_FLAT = "checkpoints/pair_cf_treatment_s1337_best.pt"
RUN_NAME = "convnextv2_atto_v1"
OFFSETS_16 = [-90, -84, -78, -72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]
LOCAL_DATASET_ROOTS = (
    WORKSPACE / "groups15-n400",
    WORKSPACE / "blender_sim" / "datasets" / "groups15_n400_s67_n400",
)


def _first_file(*cands: Path) -> Path | None:
    for p in cands:
        if p is not None and p.is_file():
            return p
    return None


def _first_complete(*cands: Path) -> Path | None:
    for p in cands:
        if p is None:
            continue
        if (p / "COMPLETE").is_file():
            return p
        if p.is_dir() and (p / "mapping.json").is_file() and (p / "F_dec.npy").is_file():
            return p
    return None


def _first_rgb(*cands: Path) -> Path | None:
    for p in cands:
        if p is None or not p.is_dir():
            continue
        if (p / "loader.py").is_file() and ((p / "COMPLETE").is_file() or (p / "metadata.json").is_file()):
            return p
        # Trial-3 letterboxed 256 frame cache (cache/frames or cache root).
        sample = next(p.glob("*/frames_uint8.npy"), None)
        if sample is not None:
            return p
        sample = next((p / "frames").glob("*/frames_uint8.npy"), None) if (p / "frames").is_dir() else None
        if sample is not None:
            return p
    return None


def parent_ckpt() -> Path:
    env = os.environ.get("STUDENT_PARENT_CKPT")
    if env:
        return Path(env)
    found = _first_file(
        TRIAL3 / PARENT_CKPT_NAME,
        TRIAL3 / PARENT_CKPT_FLAT,
        TRIAL5 / PARENT_CKPT_NAME,
        TRIAL5 / PARENT_CKPT_FLAT,
        TRIAL3 / "checkpoints" / "teacher_phase_a_best.pt",
    )
    if found is None:
        return TRIAL3 / PARENT_CKPT_NAME
    return found


def rgb_cache() -> Path:
    env = os.environ.get("STUDENT_RGB_CACHE")
    if env:
        return Path(env)
    found = _first_rgb(
        TEMP3 / "shared_rgb_cache_256_v1",
        TRIAL3 / "shared_rgb_cache_256_v1",
        TRIAL5 / "shared_rgb_cache_256_v1",
        TRIAL3 / "cache" / "frames",
        TRIAL3 / "cache",
        TRIAL5 / "cache" / "frames",
        TRIAL5 / "cache",
    )
    if found is None:
        return TRIAL3 / "cache" / "frames"
    return found


def parent_cache() -> Path:
    env = os.environ.get("STUDENT_PARENT_CACHE")
    if env:
        return Path(env)
    found = _first_complete(
        TRIAL3 / "checkpoints" / "parent_target_cache_v2",
        TRIAL5 / "checkpoints" / "parent_target_cache_v2",
        TEMP3 / "parent_target_cache_v2",
        TRIAL3 / "parent_target_cache_v2",
        TRIAL5 / "parent_target_cache_v2",
    )
    if found is None:
        return TRIAL3 / "checkpoints" / "parent_target_cache_v2"
    return found


def run_dir(name: str = RUN_NAME) -> Path:
    env = os.environ.get("STUDENT_RUN_DIR")
    if env:
        return Path(env)
    return TRIAL5 / "student_runs" / name


def stats_path() -> Path:
    env = os.environ.get("STUDENT_FDEC_STATS")
    if env:
        return Path(env)
    return TRIAL5 / "student_runs" / "fdec_channel_stats.npz"


def resolve_clip_dir(clip_id: str) -> Path | None:
    """Local dataset folder for a clip_id like sample_0000_jaywalker/collision/bg_00."""
    rel = Path(str(clip_id))
    for root in LOCAL_DATASET_ROOTS:
        cand = root / rel
        if cand.is_dir():
            return cand
    return None
