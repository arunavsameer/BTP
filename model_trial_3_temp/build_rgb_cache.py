# -*- coding: utf-8 -*-
"""
build_rgb_cache.py
------------------
Creates a compact, memory‑mapped RGB cache for all required anchors.

Output:
    model_trial_3_temp/shared_rgb_cache_v1/
        <clip_id>.npy                # uint8 [N,160,160,3] (memmap)
        index.json                   # mapping (anchor_frame, offset) → idx
        metadata.json                # provenance & stats
"""

import json
import hashlib
import numpy as np
import cv2  # OpenCV for fast video decode
from pathlib import Path
from tqdm import tqdm

# ----------------------------------------------------------------------
# Configurable constants
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "shared_rgb_cache_v1"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

TRAIN_MANIFEST = ROOT / "reports" / "student_architecture_tests" / "part1_train_manifest.json"
DEV_MANIFEST   = ROOT / "reports" / "student_architecture_tests" / "part1_dev_manifest.json"

# Offsets for the four temporal windows (in frames, relative to anchor)
OFFSET_SETS = {
    3:  [-42, -21, 0],                     # placeholder (not used directly – keep for completeness)
    5:  [-90, -66, -42, -12, 0],
    8:  [-126, -102, -78, -54, -30, -6, -2, 0],
    16: list(range(-240, 1, 16)),          # -240, -224, … , 0 (16 entries)
}
MAX_OFFSETS = sorted(set(o for vals in OFFSET_SETS.values() for o in vals))

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(p: Path) -> list[int]:
    with p.open() as f:
        data = json.load(f)
    key = "train_row_ids" if "train_row_ids" in data else "dev_row_ids"
    return data[key]


def load_teacher_index() -> dict[int, dict]:
    """Map row_id → entry from teacher_cache/index.jsonl (fast lookup)."""
    index_path = ROOT / "reports" / "student_architecture_tests" / "teacher_cache" / "index.jsonl"
    mapping = {}
    with index_path.open() as f:
        for line in f:
            rec = json.loads(line)
            mapping[rec["row_id"]] = rec
    return mapping


def frame_indices_for_anchor(anchor: int) -> set[int]:
    """All frame indices needed for a given anchor (including all offset sets)."""
    return {anchor + off for off in MAX_OFFSETS if anchor + off >= 0}


def decode_frames(video_path: Path, needed_idxs: set[int]) -> dict[int, np.ndarray]:
    """Decode only the required frames, apply optimized_v2 preprocessing.
    Returns a dict mapping frame index → RGB uint8 array of shape (160,160,3).
    Missing frames are replaced with a black placeholder and a warning is logged.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    for i in needed_idxs:
        if i >= total:
            # out‑of‑range request – use black placeholder
            frames[i] = np.zeros((160, 160, 3), dtype=np.uint8)
            continue
        # Seek to the exact frame index
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, img = cap.read()
        if not ret or img is None:
            # Fallback placeholder if reading fails
            frames[i] = np.zeros((160, 160, 3), dtype=np.uint8)
            continue
        h, w = img.shape[:2]
        scale = 256 / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h = 256 - new_h
        pad_w = 256 - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    borderType=cv2.BORDER_CONSTANT, value=0)
        start_y = (256 - 160) // 2
        start_x = (256 - 160) // 2
        cropped = padded[start_y:start_y + 160, start_x:start_x + 160]
        frames[i] = cropped.astype(np.uint8)
    cap.release()
    return frames
    """Decode only the required frames, apply optimized_v2 preprocessing.
    Returns a dict mapping frame index → RGB uint8 array of shape (160,160,3)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = {}
    for i in range(total):
        ret, img = cap.read()
        if not ret:
            break
        if i in needed_idxs:
            h, w = img.shape[:2]
            scale = 256 / max(h, w)
            new_h, new_w = int(round(h * scale)), int(round(w * scale))
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            pad_h = 256 - new_h
            pad_w = 256 - new_w
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        borderType=cv2.BORDER_CONSTANT, value=0)
            start_y = (256 - 160) // 2
            start_x = (256 - 160) // 2
            cropped = padded[start_y:start_y + 160, start_x:start_x + 160]
            frames[i] = cropped.astype(np.uint8)
    cap.release()
    return frames


def build_one_clip(clip_id: str, rows: list[dict], teacher_index: dict[int, dict]) -> None:
    """Process a single clip: decode needed frames once, write memmap, index mapping."""
    clip_dir = CACHE_ROOT / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    # Gather needed frames across all anchors in this clip
    all_needed = set()
    anchor_to_offsets = {}
    for r in rows:
        anchor = r["frame"]  # field name in teacher index is "frame"
        needed = frame_indices_for_anchor(anchor)
        anchor_to_offsets[anchor] = sorted(needed - {anchor})
        all_needed.update(needed)

    # Locate original video file – use the explicit "path" field if present
    # Teacher index entries contain "path" (absolute path) for each row.
    example_rec = rows[0]
    video_path = Path(example_rec.get("path", ""))
    if not video_path.is_file():
        # Fallback: construct from sample_id (which matches the filename without slashes)
        sample_id = example_rec.get("sample_id")
        if sample_id:
            # Files are stored under visualizations/test_videos/ as "<sample_id>_now_1s.mp4"
            video_path = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3/visualizations/test_videos") / f"{sample_id}_now_1s.mp4"
    if not video_path.is_file():
        print(f"Warning: video not found for clip_id={clip_id}, skipping this clip.")
        return

    decoded = decode_frames(video_path, all_needed)
    # Ensure every needed index has an entry (placeholder for any missing frames)
    for idx in all_needed:
        if idx not in decoded:
            decoded[idx] = np.zeros((160, 160, 3), dtype=np.uint8)


    sorted_idxs = sorted(all_needed)
    memmap_path = clip_dir / "frames.npy"
    mem = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(len(sorted_idxs), 160, 160, 3))
    idx_map = {}
    for pos, frm in enumerate(sorted_idxs):
        mem[pos] = decoded[frm]
        idx_map[frm] = pos
    mem.flush()
    del mem

    # Build anchor_offset → memmap index mapping
    anchor_offset_map = {}
    for anchor, offsets in anchor_to_offsets.items():
        anchor_offset_map[str(anchor)] = {str(off): idx_map[anchor + off] for off in offsets if (anchor + off) in idx_map}

    with (clip_dir / "index.json").open("w") as f:
        json.dump(anchor_offset_map, f, indent=2, sort_keys=True)

    meta = {
        "clip_id": clip_id,
        "video_sha256": sha256_of_file(video_path),
        "num_frames_cached": len(sorted_idxs),
        "anchors": len(rows),
        "offsets_per_anchor": {str(a): len(o) for a, o in anchor_to_offsets.items()},
        "preprocess": {
            "resize_mode": "letterbox",
            "input_size": 256,
            "final_size": 160,
            "normalize": False,
        },
    }
    with (clip_dir / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)


def main():
    train_ids = load_manifest(TRAIN_MANIFEST)
    dev_ids   = load_manifest(DEV_MANIFEST)
    teacher_index = load_teacher_index()

    # Filter out any row ids that are missing from the teacher index (should be rare)
    train_ids = [i for i in train_ids if i in teacher_index]
    dev_ids   = [i for i in dev_ids   if i in teacher_index]
    if len(train_ids) < len(load_manifest(TRAIN_MANIFEST)):
        print(f"Warning: {len(load_manifest(TRAIN_MANIFEST)) - len(train_ids)} training rows missing from teacher cache and will be skipped.")
    if len(dev_ids)   < len(load_manifest(DEV_MANIFEST)):
        print(f"Warning: {len(load_manifest(DEV_MANIFEST)) - len(dev_ids)} dev rows missing from teacher cache and will be skipped.")

    all_rows = [teacher_index[i] for i in train_ids + dev_ids]
    rows_by_clip = {}
    for rec in all_rows:
        rows_by_clip.setdefault(rec["clip_id"], []).append(rec)

    for clip_id, rows in tqdm(rows_by_clip.items(), desc="Building RGB cache"):
        clip_dir = CACHE_ROOT / clip_id
        if (clip_dir / "frames.npy").exists():
            continue
        build_one_clip(clip_id, rows, teacher_index)

    overall = {
        "total_clips": len(rows_by_clip),
        "total_anchors": sum(len(v) for v in rows_by_clip.values()),
        "offset_sets": OFFSET_SETS,
        "source_repo": "model_trial_3 (read‑only)",
    }
    with (CACHE_ROOT / "metadata.json").open("w") as f:
        json.dump(overall, f, indent=2)

if __name__ == "__main__":
    main()
