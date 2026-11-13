#!/usr/bin/env python3
"""Build 256×256 uint8 RGB cache for student_lat from original videos (not from 160 cache)."""

from __future__ import annotations

import json
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MT3 = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3")
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
OUT = TEMP / "shared_rgb_cache_256_v1"
V2 = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3/shared_rgb_cache_v2")
PARENT = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3/parent_target_cache_v2")
REPORTS = TEMP / "reports"
SIZE = 256
OFFSETS_16 = [
    -90, -84, -78, -72, -66, -60, -54, -48,
    -42, -36, -30, -24, -18, -12, -6, 0,
]
sys.path.insert(0, str(MT3))
sys.path.insert(0, str(OUT))

from collision_jepa3.config import Config  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.preprocessing import IMAGENET_MEAN, IMAGENET_STD, letterbox_rgb, signature_from_cfg  # noqa: E402
from collision_jepa3.data.video_reader import decode_indices_rgb  # noqa: E402
from loader import RGBCache256  # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def needed_from_parent() -> dict[str, list[int]]:
    rows = [json.loads(l) for l in (PARENT / "index.jsonl").open(encoding="utf-8")]
    by: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        t = int(r["frame"])
        for o in OFFSETS_16:
            by[r["clip_id"]].add(max(0, t + int(o)))
    return {cid: sorted(fs) for cid, fs in by.items()}


def verify_v2_inventory(needed: dict[str, list[int]]) -> dict:
    mismatch = 0
    for cid, frames in needed.items():
        idxp = V2 / cid / "index.json"
        if not idxp.is_file():
            mismatch += 1
            continue
        if [int(x) for x in json.loads(idxp.read_text())] != frames:
            mismatch += 1
    if mismatch:
        raise RuntimeError(f"v2 inventory mismatch clips={mismatch}")
    return {"v2_clips": len(needed), "v2_match": True, "unique_frames": sum(len(v) for v in needed.values())}


def estimate(needed: dict[str, list[int]]) -> dict:
    n = sum(len(v) for v in needed.values())
    bytes_ = n * SIZE * SIZE * 3
    return {
        "unique_clips": len(needed),
        "unique_frames": n,
        "bytes": bytes_,
        "gib": bytes_ / (1024**3),
        "vs_160_factor": (SIZE / 160) ** 2,
    }


def _write_one(args: tuple) -> tuple[str, int, bool]:
    cid, video_path, frames, dest = args
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    npy = dest / "frames.npy"
    tmp = dest / "frames.npy.tmp"
    fi = dest / "frame_indices.npy"
    if npy.is_file() and fi.is_file():
        idx = np.load(fi)
        if list(idx) == frames and npy.stat().st_size == len(frames) * SIZE * SIZE * 3:
            return cid, len(frames), True
    packed = np.empty((len(frames), SIZE, SIZE, 3), dtype=np.uint8)
    import cv2

    want = {int(f): j for j, f in enumerate(frames)}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed {video_path}")
    filled = 0
    i = 0
    last = None
    try:
        while filled < len(want):
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                packed[want[i]] = letterbox_rgb(rgb, SIZE, 0)
                last = rgb
                filled += 1
            i += 1
    finally:
        cap.release()
    if filled < len(want):
        if last is None:
            raise RuntimeError(f"decoded 0 {cid}")
        for f, j in want.items():
            if f >= i:
                packed[j] = letterbox_rgb(last, SIZE, 0)
    np.save(tmp, packed)
    tmp.replace(npy)
    np.save(fi, np.asarray(frames, dtype=np.int32))
    (dest / "index.json").write_text(json.dumps(frames), encoding="utf-8")
    return cid, len(frames), False


def validate(cfg, needed: dict[str, list[int]], recs: dict, t0: float, peak_rss: float) -> dict:
    from loader import IMAGENET_MEAN as M, IMAGENET_STD as S

    cache = RGBCache256(OUT)
    rows = [json.loads(l) for l in (PARENT / "index.jsonl").open(encoding="utf-8")]
    missing = 0
    order_ok = 0
    for r in rows:
        try:
            u = cache.get_frames_uint8(r["clip_id"], int(r["frame"]), OFFSETS_16)
        except Exception as e:
            missing += 1
            if missing <= 3:
                log(f"coverage miss {r['clip_id']} t={r['frame']}: {e}")
            continue
        if u.shape != (16, SIZE, SIZE, 3):
            missing += 1
            continue
        order_ok += 1
    if missing:
        raise RuntimeError(f"window coverage fail missing={missing}")

    rng = np.random.default_rng(1337)
    rec_list = [recs[cid] for cid in needed if cid in recs]
    diffs = []
    probes = []
    while len(probes) < 100:
        rec = rec_list[int(rng.integers(0, len(rec_list)))]
        frames = needed[rec.clip_id]
        fi = int(frames[int(rng.integers(0, len(frames)))])
        probes.append((rec, fi))
    for rec, fi in probes:
        raw, _ = decode_indices_rgb(rec.video_path, [fi])
        live = letterbox_rgb(np.asarray(raw[0]), SIZE, 0)
        cached = cache.get_frames_uint8(rec.clip_id, fi, [0])[0]
        diffs.append(float(np.max(np.abs(live.astype(np.int16) - cached.astype(np.int16)))))
    max_diff = max(diffs) if diffs else None
    if max_diff is None or max_diff > 0:
        raise RuntimeError(f"pixel parity failed max_abs={max_diff} n={len(diffs)}")

    u = cache.get_frames_uint8(probes[0][0].clip_id, probes[0][1], [0])
    n01 = np.transpose(u.astype(np.float32), (0, 3, 1, 2)) / 255.0
    norm = cache.get_frames(probes[0][0].clip_id, probes[0][1], [0], input_size=256, normalize=True)
    expect = (n01 - M) / S
    if float(np.max(np.abs(norm - expect))) > 1e-6:
        raise RuntimeError("imagenet norm mismatch")

    # DataLoader workers
    from torch.utils.data import DataLoader, Dataset

    class _DS(Dataset):
        def __init__(self):
            self.rows = rows[:64]
            self.c = RGBCache256(OUT)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            r = self.rows[i]
            return self.c.get_frames(r["clip_id"], int(r["frame"]), OFFSETS_16)

    t1 = time.perf_counter()
    n_ret = 0
    for r in rows[:200]:
        cache.get_frames_uint8(r["clip_id"], int(r["frame"]), OFFSETS_16)
        n_ret += 1
    retr_s = (time.perf_counter() - t1) / max(1, n_ret)
    loader = DataLoader(_DS(), batch_size=4, num_workers=2)
    n_batches = 0
    for batch in loader:
        assert tuple(batch.shape[-3:]) == (3, SIZE, SIZE)
        n_batches += 1

    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "selection", "test")}
    inv = {
        "unique_clips": len({r["clip_id"] for r in rows}),
        "unique_sample_ids": len({r["sample_id"] for r in rows}),
        "unique_frames": sum(len(v) for v in needed.values()),
        "windows": {s: len(splits[s]) for s in splits},
        "sample_ids_by_split": {s: len({r["sample_id"] for r in splits[s]}) for s in splits},
        "clips_by_split": {s: len({r["clip_id"] for r in splits[s]}) for s in splits},
        "no_sample_overlap": not (
            {r["sample_id"] for r in splits["train"]} & {r["sample_id"] for r in splits["selection"]}
            or {r["sample_id"] for r in splits["train"]} & {r["sample_id"] for r in splits["test"]}
            or {r["sample_id"] for r in splits["selection"]} & {r["sample_id"] for r in splits["test"]}
        ),
        "pixel_parity_n": len(diffs),
        "pixel_parity_max_abs": max_diff,
        "imagenet_norm_ok": True,
        "windows_covered": order_ok,
        "dataloader_worker_batches": n_batches,
        "random_retrieval_sec_per_window": retr_s,
        "prep_sec": time.perf_counter() - t0,
        "peak_rss_gb_builder": peak_rss,
        "disk_bytes": sum(p.stat().st_size for p in OUT.rglob("frames.npy")),
        "size": SIZE,
        "offsets": OFFSETS_16,
        "source": "original videos, letterbox_rgb(256) same as student_lat/TeacherDataset",
        "not_upscaled_from_160": True,
    }
    return inv


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.load(str(MT3 / "configs/default.yaml"))
    sig = signature_from_cfg(cfg)
    if sig.input_size != 256 or sig.resize_mode != "letterbox":
        raise SystemExit(f"unexpected student_lat preprocess {sig}")
    recs = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    needed = needed_from_parent()
    v2 = verify_v2_inventory(needed)
    est = estimate(needed)
    log(f"estimate {est} v2={v2}")
    import shutil

    free = shutil.disk_usage(OUT.resolve()).free
    if free < est["bytes"] * 1.05:
        raise SystemExit(f"not enough disk free={free} need={est['bytes']}")
    (OUT / "estimate.json").write_text(json.dumps({"estimate": est, "v2": v2, "free_bytes": free}, indent=2), encoding="utf-8")

    jobs = []
    skip = 0
    for cid, frames in sorted(needed.items()):
        rec = recs[cid]
        d = OUT / cid
        npy, fi = d / "frames.npy", d / "frame_indices.npy"
        if npy.is_file() and fi.is_file() and npy.stat().st_size == len(frames) * SIZE * SIZE * 3:
            skip += 1
            continue
        jobs.append((cid, rec.video_path, frames, str(d)))
    log(f"skip_complete={skip} todo={len(jobs)}")
    t0 = time.perf_counter()
    import resource

    if jobs:
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = [ex.submit(_write_one, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                cid, nf, skipped = fut.result()
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    log(f"rgb256 {done}/{len(jobs)} sec={time.perf_counter()-t0:.1f}")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    inv = validate(cfg, needed, recs, t0, peak_rss)
    (REPORTS / "shared_rgb_cache_256_inventory.json").write_text(json.dumps(inv, indent=2), encoding="utf-8")
    md = f"""# shared_rgb_cache_256_v1

student_lat-compatible 256×256 uint8 RGB cache. Not upscaled from 160.

- Unique clips: {inv['unique_clips']}
- Unique sample IDs: {inv['unique_sample_ids']}
- Unique frames: {inv['unique_frames']}
- Windows: {inv['windows']}
- Disk bytes: {inv['disk_bytes']} ({inv['disk_bytes']/1024**3:.2f} GiB)
- Prep seconds: {inv['prep_sec']:.1f}
- Pixel parity max abs (100 frames vs letterbox_rgb on original decode): {inv['pixel_parity_max_abs']}
- ImageNet load-time normalization: ok
- Random retrieval s/window: {inv['random_retrieval_sec_per_window']:.6f}
- DataLoader worker batches: {inv['dataloader_worker_batches']}
- Preprocess: OpenCV BGR→RGB, `letterbox_rgb(size=256, pad=0)` matching `TeacherDataset._load_input_clip` / `signature_from_cfg` (`input_size=256`).
"""
    (REPORTS / "shared_rgb_cache_256_report.md").write_text(md, encoding="utf-8")
    (OUT / "metadata.json").write_text(json.dumps({"root": str(OUT), **est, **inv}, indent=2), encoding="utf-8")
    (OUT / "COMPLETE").write_text("ok\n", encoding="utf-8")
    log(f"DONE {inv}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        (OUT / ".failed").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        raise
