#!/usr/bin/env python3
"""Build RAID RGB v2 + parent F_dec/H caches for Pair-CF-matched LatentMatch V3."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEMP = Path("/home/asrivastava/VJEPA2_BTP/BTP/model_trial_3_temp")
RAID = Path("/raid/asrivastava/VJEPA2_BTP/model_trial_3")
RGB_RAID = RAID / "shared_rgb_cache_v2"
PARENT_RAID = RAID / "parent_target_cache_v2"
sys.path.insert(0, str(ROOT))

from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.cache import TokenCache, _clip_frames_uint8, token_cache_id, token_fingerprint  # noqa: E402
from collision_jepa3.data.heatmap_annotations import load_heatmaps, targets_for_anchor  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.pairing import compute_ambiguity_table  # noqa: E402
from collision_jepa3.data.preprocessing import letterbox_rgb, signature_from_cfg, uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import FRAME_OFFSETS, HORIZONS_FRAMES, input_indices  # noqa: E402
from collision_jepa3.data.video_reader import decode_all_rgb  # noqa: E402
from collision_jepa3.evaluation.metrics import classify_heatmap  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.student_research.parent import load_pair_cf_parent, parent_forward, sha256_file  # noqa: E402
from collision_jepa3.student_research.parent_matched_contract import OFFSETS_16, write_sampling_contract  # noqa: E402
from collision_jepa3.student_research.student_final.targets import build_group_mean_gt_array  # noqa: E402
from collision_jepa3.utils.device import resolve_device_safe  # noqa: E402

PARENT_CKPT = ROOT / "checkpoints/pair_cf_treatment_s1337/teacher_phase_a_best.pt"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        dst.unlink()
    dst.symlink_to(src)


def needed_frames_by_clip(rows: list[dict]) -> dict[str, list[int]]:
    by: dict[str, set[int]] = defaultdict(set)
    for r in rows:
        t = int(r["frame"])
        for o in OFFSETS_16:
            by[r["clip_id"]].add(max(0, t + int(o)))
    return {cid: sorted(fs) for cid, fs in by.items()}


def _write_one_rgb(args: tuple) -> tuple[str, int, bool]:
    import cv2

    cid, video_path, frames, dest = args
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    npy = dest / "frames.npy"
    tmp = dest / "frames.npy.tmp"
    idxp = dest / "index.json"
    if npy.is_file() and idxp.is_file():
        idx = json.loads(idxp.read_text())
        if list(idx) == frames and npy.stat().st_size == len(frames) * 160 * 160 * 3:
            return cid, len(frames), True
    want = {int(f): j for j, f in enumerate(frames)}
    packed = np.empty((len(frames), 160, 160, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed {video_path}")
    filled = 0
    i = 0
    last_rgb = None
    try:
        while filled < len(want):
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                packed[want[i]] = letterbox_rgb(rgb, 160)
                last_rgb = rgb
                filled += 1
            i += 1
    finally:
        cap.release()
    if filled < len(want):
        if last_rgb is None:
            raise RuntimeError(f"decoded 0 frames {cid}")
        for fi, j in want.items():
            if fi >= i:
                packed[j] = letterbox_rgb(last_rgb, 160)
    packed.tofile(tmp)
    tmp.replace(npy)
    idxp.write_text(json.dumps(frames), encoding="utf-8")
    return cid, len(frames), False


def build_rgb(cfg, rows: list[dict], workers: int = 8) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    RGB_RAID.mkdir(parents=True, exist_ok=True)
    recs = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    needed = needed_frames_by_clip(rows)
    jobs = []
    n_skip = n_frames = 0
    for cid, frames in sorted(needed.items()):
        d = RGB_RAID / cid
        npy = d / "frames.npy"
        idxp = d / "index.json"
        if npy.is_file() and idxp.is_file():
            idx = json.loads(idxp.read_text())
            if list(idx) == frames and npy.stat().st_size == len(frames) * 160 * 160 * 3:
                n_skip += 1
                n_frames += len(frames)
                continue
        jobs.append((cid, recs[cid].video_path, frames, str(d)))
    t0 = time.perf_counter()
    n_ok = n_skip
    _log(f"rgb skip_complete={n_skip} todo={len(jobs)} workers={workers}")
    if jobs:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = [ex.submit(_write_one_rgb, j) for j in jobs]
            done = 0
            for fut in as_completed(futs):
                cid, nf, skipped = fut.result()
                n_ok += 1
                n_frames += nf
                done += 1
                if done % 20 == 0 or done == len(jobs):
                    _log(f"rgb {done}/{len(jobs)} new frames_total={n_frames} sec={time.perf_counter()-t0:.1f}")
    meta = {
        "clips": n_ok,
        "unique_frames": n_frames,
        "bytes": n_frames * 160 * 160 * 3,
        "root": str(RGB_RAID),
        "offsets": OFFSETS_16,
    }
    (RGB_RAID / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (RGB_RAID / "COMPLETE").write_text("ok\n", encoding="utf-8")
    _symlink(RGB_RAID, TEMP / "shared_rgb_cache_v2")
    return meta


def _attach_gt_amb(cfg, rows: list[dict]) -> tuple[list[dict], np.ndarray]:
    recs = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
    rec_by = {r.clip_id: r for r in recs}
    n_frames = int(cfg.get("data.expected_frames", 450))
    horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
    names = list(cfg.get("sampling.horizon_names", ["1s", "1_5s", "2s"]))
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    heatmaps = {}
    for r in rows:
        cid = r["clip_id"]
        if cid not in heatmaps and cid in rec_by:
            heatmaps[cid] = load_heatmaps(rec_by[cid].heatmap_json_path, expected_frames=n_frames, grid=5)
    recs_used = [rec_by[cid] for cid in heatmaps]
    amb = compute_ambiguity_table(
        heatmaps,
        recs_used,
        fork_frame=int(cfg.get("sampling.fork_frame", 365)),
        n_frames=n_frames,
        stride=15,
        offsets=offsets,
        horizons=horizons,
        horizon_names=names,
        heatmap_tol=float(cfg.get("sampling.heatmap_disagree_tol", 0.02)),
        mask_ambiguous=True,
        allow_partial_horizons=True,
    )
    raw = np.zeros((len(rows), 5, 5), dtype=np.float32)
    for i, row in enumerate(rows):
        cid = row["clip_id"]
        t = int(row["frame"])
        flags = amb.get((cid, t), {n: False for n in names})
        row["amb_1s"] = bool(flags.get("1s", False))
        row["amb_1_5s"] = bool(flags.get("1_5s", False))
        row["amb_2s"] = bool(flags.get("2s", False))
        row["row_id"] = i
        tg = targets_for_anchor(heatmaps[cid], t, horizons, allow_missing=True)
        h = tg.get("h_1s")
        raw[i] = np.zeros((5, 5), np.float32) if h is None else np.asarray(h, dtype=np.float32)
    gt, meta = build_group_mean_gt_array(cfg, rows, raw_gt=raw)
    _log(f"gt_meta {meta}")
    return rows, gt


def build_parent(cfg, rows: list[dict], gt: np.ndarray, device: str, batch: int) -> dict:
    PARENT_RAID.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    F = np.memmap(PARENT_RAID / "F_dec.npy", dtype=np.float16, mode="w+", shape=(n, 128, 16, 16))
    H = np.memmap(PARENT_RAID / "H_teacher.npy", dtype=np.float16, mode="w+", shape=(n, 5, 5))
    Z = np.memmap(PARENT_RAID / "Z_t.npy", dtype=np.float16, mode="w+", shape=(n, 32, 5, 5))
    F[:] = np.nan
    H[:] = np.nan
    Z[:] = np.nan
    packs = {
        "H": np.full((n, 5, 5), np.nan, np.float32),
        "Z": np.full((n, 32, 5, 5), np.nan, np.float32),
        "F_dec": np.full((n, 128, 16, 16), np.nan, np.float32),
        "peak": np.full((n,), np.nan, np.float32),
    }
    tier = [""] * n
    warning = cfg.get("warning") or {}
    model, blob = load_pair_cf_parent(cfg, PARENT_CKPT, device=device)
    fps_try = [
        token_cache_id(cfg, dummy=False, stride=6),
        token_cache_id(cfg, dummy=False),
        token_fingerprint(cfg, dummy=False),
    ]
    tok_caches = [TokenCache(cfg.path("paths.cache_root"), fp) for fp in fps_try]
    records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    sig = signature_from_cfg(cfg)
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    by_clip: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_clip[row["clip_id"]].append(i)
    filled_cache = filled_enc = 0
    encoder = None
    frame_mem: dict[str, np.ndarray] = {}
    import torch

    for ci, (cid, idxs) in enumerate(by_clip.items(), start=1):
        cache_rows, cache_toks, need_encode = [], [], []
        for i in idxs:
            t = int(rows[i]["frame"])
            vec = None
            for cache in tok_caches:
                try:
                    arr, mapping = cache.load_clip(cid, mmap=True)
                except FileNotFoundError:
                    continue
                if t not in mapping:
                    continue
                vec = np.asarray(arr[mapping[t]], dtype=np.float32)
                break
            if vec is not None:
                cache_rows.append(i)
                cache_toks.append(vec)
            else:
                need_encode.append(i)
        if cache_toks:
            x = torch.from_numpy(np.stack(cache_toks, axis=0)).to(device)
            for start in range(0, len(cache_rows), batch):
                sl = slice(start, start + batch)
                out = parent_forward(x[sl], model)
                from collision_jepa3.student_research.teacher_cache import _write_pack

                _write_pack(packs, tier, rows, warning, cache_rows[sl], out)
                filled_cache += len(cache_rows[sl])
        if need_encode:
            if encoder is None:
                encoder = CollisionTeacher3(cfg, dummy_backbone=False).to(device).eval()
            if cid not in frame_mem:
                rec = records.get(cid)
                if rec is None:
                    continue
                frame_mem[cid] = _clip_frames_uint8(rec, cfg, sig)
            frames = frame_mem[cid]
            for start in range(0, len(need_encode), batch):
                chunk = need_encode[start : start + batch]
                windows = [
                    uint8_to_nchw01(
                        np.stack([np.asarray(frames[k]) for k in input_indices(int(rows[rid]["frame"]), offsets)], axis=0)
                    )
                    for rid in chunk
                ]
                clips = torch.from_numpy(np.stack(windows, axis=0)).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    tokens = encoder.encode_tokens(clips)
                out = parent_forward(tokens, model)
                from collision_jepa3.student_research.teacher_cache import _write_pack

                _write_pack(packs, tier, rows, warning, chunk, out)
                filled_enc += len(chunk)
            if cid in frame_mem:
                del frame_mem[cid]
        if ci % 16 == 0 or ci == len(by_clip):
            _log(f"parent clips {ci}/{len(by_clip)} cache={filled_cache} enc={filled_enc}")
    F[:] = packs["F_dec"].astype(np.float16)
    H[:] = packs["H"].astype(np.float16)
    Z[:] = packs["Z"].astype(np.float16)
    F.flush()
    H.flush()
    Z.flush()
    np.save(PARENT_RAID / "gt_1s.npy", gt.astype(np.float16))
    mapping = {f"{r['clip_id']}::{int(r['frame'])}": i for i, r in enumerate(rows)}
    (PARENT_RAID / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    with (PARENT_RAID / "index.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    n_train = sum(1 for r in rows if r["split"] == "train")
    meta = {
        "checkpoint": str(PARENT_CKPT),
        "checkpoint_sha256": sha256_file(PARENT_CKPT),
        "n_rows": n,
        "num_training_anchors": n,
        "n_train": n_train,
        "filled_token_cache": filled_cache,
        "filled_encode": filled_enc,
        "missing": int(n - filled_cache - filled_enc),
        "tensor_shapes": {"H": [5, 5], "F_dec": [128, 16, 16]},
        "ckpt_epoch": blob.get("epoch"),
    }
    (PARENT_RAID / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if int(meta["missing"]) != 0:
        raise RuntimeError(f"parent cache incomplete missing={meta['missing']}")
    (PARENT_RAID / "COMPLETE").write_text("ok\n", encoding="utf-8")
    _symlink(PARENT_RAID, TEMP / "parent_target_cache_v2")
    return meta


def replay_decoder(cfg, rows: list[dict], device: str, n_check: int = 100) -> dict:
    import torch
    from collision_jepa3.student_research.cache_access import ParentTargetCache

    parent = ParentTargetCache(PARENT_RAID)
    model, _ = load_pair_cf_parent(cfg, PARENT_CKPT, device=device)
    fps_try = [
        token_cache_id(cfg, dummy=False, stride=6),
        token_cache_id(cfg, dummy=False),
        token_fingerprint(cfg, dummy=False),
    ]
    tok_caches = [TokenCache(cfg.path("paths.cache_root"), fp) for fp in fps_try]
    rng = np.random.default_rng(1337)
    train_idx = [i for i, r in enumerate(rows) if r["split"] == "train"]
    pick = rng.choice(train_idx, size=min(n_check, len(train_idx)), replace=False)
    diffs = []
    used = 0
    for i in pick:
        row = rows[int(i)]
        cid, t = row["clip_id"], int(row["frame"])
        vec = None
        for cache in tok_caches:
            try:
                arr, mapping = cache.load_clip(cid, mmap=True)
            except FileNotFoundError:
                continue
            if t not in mapping:
                continue
            vec = np.asarray(arr[mapping[t]], dtype=np.float32)
            break
        if vec is None:
            continue
        out = parent_forward(torch.from_numpy(vec[None]).to(device), model)
        cached = parent.get_parent_targets(cid, t)
        if cached is None:
            continue
        a = out["F_pre_decoder"][0].float().cpu().numpy()
        diffs.append(float(np.max(np.abs(a - cached["F_dec"]))))
        used += 1
    rep = {
        "checked": used,
        "max_abs": max(diffs) if diffs else None,
        "ok": used >= min(100, n_check) and max(diffs or [0]) < 2e-2,
        "note": "fp16 cache vs fp32 replay",
    }
    (PARENT_RAID / "decoder_replay.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if not rep["ok"]:
        raise RuntimeError(f"decoder replay failed {rep}")
    return rep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="all", choices=["contract", "rgb", "parent", "all"])
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--rgb-workers", type=int, default=8)
    add_config_args(parser)
    args = parser.parse_args()
    cfg = load_cfg(args)
    RAID.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)
    contract_path = RAID / "sampling_contract.json"
    if args.stage in ("contract", "all") or not contract_path.is_file():
        info = write_sampling_contract(cfg, contract_path)
        _log(f"contract {info}")
        _symlink(contract_path, TEMP / "sampling_contract.json")
    blob = json.loads(contract_path.read_text())
    rows = blob["rows"]
    if args.stage in ("rgb", "all"):
        meta = build_rgb(cfg, rows, workers=args.rgb_workers)
        _log(f"rgb {meta}")
    if args.stage in ("parent", "all"):
        device = resolve_device_safe(str(args.device))
        rows, gt = _attach_gt_amb(cfg, rows)
        meta = build_parent(cfg, rows, gt, device, args.batch)
        _log(f"parent {meta}")
        replay = replay_decoder(cfg, rows, device)
        _log(f"replay {replay}")
    _log("DONE caches")


if __name__ == "__main__":
    main()
