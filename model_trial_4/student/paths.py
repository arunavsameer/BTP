"""Read-only Trial-3 parent artifacts; Trial-4 write roots."""

from __future__ import annotations

import os
from pathlib import Path

TRIAL4 = Path(__file__).resolve().parent.parent
TRIAL3 = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3")
TEMP3 = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RAID3 = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
RAID3_TEMP = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3_temp")
RAID4 = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_4")

EXPECTED_DECODER_HASH = "77f78c8714667617"
PARENT_CKPT_NAME = "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt"
RUN_NAME = "lat_paircf_v1"
OFFSETS_16 = [-90, -84, -78, -72, -66, -60, -54, -48, -42, -36, -30, -24, -18, -12, -6, 0]


def _first_complete(*cands: Path) -> Path | None:
    for p in cands:
        if p is None:
            continue
        if (p / "COMPLETE").is_file():
            return p
        if p.is_dir() and (p / "mapping.json").is_file() and (p / "F_dec.npy").is_file():
            return p
    return None


def parent_ckpt() -> Path:
    env = os.environ.get("STUDENT_PARENT_CKPT")
    if env:
        return Path(env)
    return TRIAL3 / PARENT_CKPT_NAME


def rgb_cache() -> Path:
    env = os.environ.get("STUDENT_RGB_CACHE")
    if env:
        return Path(env)
    found = _first_complete(
        TEMP3 / "shared_rgb_cache_256_v1",
        RAID3_TEMP / "shared_rgb_cache_256_v1",
        RAID3 / "shared_rgb_cache_256_v1",
    )
    if found is None:
        return TEMP3 / "shared_rgb_cache_256_v1"
    return found


def parent_cache() -> Path:
    env = os.environ.get("STUDENT_PARENT_CACHE")
    if env:
        return Path(env)
    found = _first_complete(
        TEMP3 / "parent_target_cache_v2",
        RAID3 / "parent_target_cache_v2",
        TRIAL4 / "parent_target_cache_v2",
    )
    if found is None:
        return RAID3 / "parent_target_cache_v2"
    return found


def run_dir(name: str = RUN_NAME) -> Path:
    env = os.environ.get("STUDENT_RUN_DIR")
    if env:
        return Path(env)
    return TRIAL4 / "student_runs" / name


def z16_cache() -> Path:
    env = os.environ.get("STUDENT_Z16_CACHE")
    if env:
        return Path(env)
    return RAID4 / "z16_sidecar_v1"
