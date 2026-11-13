# -*- coding: utf-8 -*-
"""
build_parent_target_cache.py
---------------------------
Creates a compact cache of only the teacher latent (F_dec) and heatmap (H_teacher)
for *training* anchors.

Outputs:
    model_trial_3_temp/parent_target_cache_v1/
        F_dec.npy          # FP16 [N_train,128,16,16] (memmap)
        H_teacher.npy      # FP16 [N_train,5,5]
        mapping.json       # (clip_id, anchor_frame) → row index
        metadata.json
"""

import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "parent_target_cache_v1"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST = ROOT / "reports" / "student_architecture_tests" / "part1_train_manifest.json"
TEACHER_INDEX = ROOT / "reports" / "student_architecture_tests" / "teacher_cache" / "index.jsonl"
PARENT_CKPT = ROOT / "checkpoints" / "pair_cf_treatment_s1337" / "teacher_phase_a_best.pt"

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl_index(p: Path) -> dict[int, dict]:
    mapping = {}
    with p.open() as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["row_id"]] = rec
    return mapping


def main():
    # Load training row IDs
    with TRAIN_MANIFEST.open() as f:
        train_ids = json.load(f)["train_row_ids"]

    teacher_index = load_jsonl_index(TEACHER_INDEX)
    rows = [teacher_index[i] for i in train_ids]

    # Load frozen parent decoder (weights only)
    from collision_jepa3.student_research.fdec_student import build_parent_decoder
    # Load a minimal configuration required by the parent decoder
    from collision_jepa3.config import Config, DEFAULT_CONFIG_PATH
    cfg = Config.load(DEFAULT_CONFIG_PATH)
    decoder = build_parent_decoder(cfg, device="cpu")  # decoder is frozen internally

    # Prepare memmaps
    N = len(rows)
    fdec_mem = np.memmap(CACHE_ROOT / "F_dec.npy", dtype=np.float16, mode="w+", shape=(N, 128, 16, 16))
    h_mem   = np.memmap(CACHE_ROOT / "H_teacher.npy", dtype=np.float16, mode="w+", shape=(N, 5, 5))

    mapping = {}
    # Original teacher cache (contains all rows, not just training)
    orig_npz = ROOT / "reports" / "student_architecture_tests" / "teacher_cache_train_pilot" / "teacher_pred.npz"
    data = np.load(orig_npz)
    for idx, rec in enumerate(tqdm(rows, desc="Generating parent targets")):
        row_id = rec["row_id"]
        fdec = data["F_dec"][row_id].astype(np.float16)
        h    = data["H"][row_id].astype(np.float16)
        fdec_mem[idx] = fdec
        h_mem[idx]    = h
        mapping[f"{rec['clip_id']}::{rec['frame']}"] = idx

    fdec_mem.flush(); h_mem.flush()
    del fdec_mem, h_mem

    # Write mapping & metadata
    with (CACHE_ROOT / "mapping.json").open("w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    meta = {
        "parent_checkpoint": str(PARENT_CKPT),
        "parent_checkpoint_sha256": sha256_of_file(PARENT_CKPT),
        "num_training_anchors": N,
        "shapes": {
            "F_dec": [N, 128, 16, 16],
            "H_teacher": [N, 5, 5],
        },
        "dtype": "float16",
        "preprocess": {"optimized_v2": True, "normalize": False},
    }
    with (CACHE_ROOT / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()

