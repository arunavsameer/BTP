"""Dense teacher cache for Pair-CF parent (stride-6 anchors, train+val)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from collision_jepa3.data.cache import TokenCache, _clip_frames_uint8, token_cache_id, token_fingerprint
from collision_jepa3.data.heatmap_annotations import load_heatmaps, targets_for_anchor
from collision_jepa3.data.manifest import load_manifest
from collision_jepa3.data.pairing import compute_ambiguity_table
from collision_jepa3.data.preprocessing import signature_from_cfg, uint8_to_nchw01
from collision_jepa3.data.sampling import FRAME_OFFSETS, HORIZONS_FRAMES, input_indices, iter_anchors
from collision_jepa3.data.splits import load_splits
from collision_jepa3.evaluation.labels import row_scalars, true_side_for_group
from collision_jepa3.evaluation.metrics import classify_heatmap
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.training.engine import teacher_forward

from .parent import load_pair_cf_parent, parent_forward, resolve_parent_ckpt, sha256_file


CACHE_FIELDS = [
    "row_id", "clip_id", "sample_id", "frame", "fps", "split",
    "scenario_group_id", "background_pair_id", "background_id", "outcome_group_id",
    "pre_fork", "anchor_class", "warning_tier", "warning_peak",
]


def _build_rows(cfg, splits: list[str], stride: int) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    records = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
    split_map = load_splits(cfg.path("paths.split_file"))
    n_frames = int(cfg.get("data.expected_frames", 450))
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
    names = list(cfg.get("sampling.horizon_names", ["1s", "1_5s", "2s"]))
    anchors = list(iter_anchors(n_frames, stride, offsets, horizons, allow_partial_horizons=True))
    rows: list[dict] = []
    gt = {"gt_1s": []}
    warning = cfg.get("warning") or {}
    for split in splits:
        groups = set(split_map.get(split, []))
        recs = [r for r in records if r.scenario_group_id in groups]
        heatmaps = {r.clip_id: load_heatmaps(r.heatmap_json_path, expected_frames=n_frames, grid=5) for r in recs}
        amb = compute_ambiguity_table(
            heatmaps, recs,
            fork_frame=int(cfg.get("sampling.fork_frame", 365)),
            n_frames=n_frames, stride=stride, offsets=offsets, horizons=horizons,
            horizon_names=names, heatmap_tol=float(cfg.get("sampling.heatmap_disagree_tol", 0.02)),
            mask_ambiguous=True, allow_partial_horizons=True,
        )
        collision_of = {
            r.outcome_group_id: r.clip_id
            for r in recs
            if r.outcome_type == "collision" and r.outcome_group_id
        }
        true_side: dict[str, str] = {}
        for gid, cid in collision_of.items():
            rec = next(x for x in recs if x.clip_id == cid)
            true_side[gid] = true_side_for_group(heatmaps[cid], rec.scenario_type, warning)
        for rec in recs:
            fps = float(rec.fps or cfg.get("data.fps_expected", 30))
            side = true_side.get(rec.outcome_group_id or "", "center")
            for t in anchors:
                tg = targets_for_anchor(heatmaps[rec.clip_id], t, horizons, allow_missing=True)
                flags = amb.get((rec.clip_id, int(t)), {n: False for n in names})
                scalar = row_scalars(
                    clip_id=rec.clip_id, sample_id=rec.sample_id, frame=int(t), fps=fps,
                    outcome_type=rec.outcome_type, fork_frame=int(rec.fork_frame or 365),
                    hit_s=rec.hit_s, scenario_type=rec.scenario_type, true_side=side,
                    background_id=rec.background_id,
                    background_pair_id=rec.background_pair_id or "",
                    outcome_group_id=rec.outcome_group_id or "",
                    amb=flags,
                )
                scalar["row_id"] = len(rows)
                scalar["split"] = split
                scalar["scenario_group_id"] = rec.scenario_group_id
                rows.append(scalar)
                gt["gt_1s"].append(tg["h_1s"])
    stacked = {k: np.stack(v, axis=0).astype(np.float16) for k, v in gt.items()}
    meta = {
        "stride_frames": stride,
        "n_rows": len(rows),
        "splits": splits,
        "anchors_per_clip": len(anchors),
        "preprocess_parent": signature_from_cfg(cfg).as_dict(),
    }
    return rows, stacked, meta


@torch.no_grad()
def fill_teacher_cache(
    cfg,
    out_dir: Path,
    *,
    ckpt: Path,
    device: str,
    batch: int = 16,
    splits: tuple[str, ...] = ("train", "val"),
    stride: int = 6,
    encode_missing: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, gt, meta = _build_rows(cfg, list(splits), stride)
    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    np.savez_compressed(out_dir / "gt_heatmaps.npz", **gt)
    model, blob = load_pair_cf_parent(cfg, ckpt, device=device)
    n = len(rows)
    packs = {
        "H": np.full((n, 5, 5), np.nan, dtype=np.float32),
        "Z": np.full((n, 32, 5, 5), np.nan, dtype=np.float32),
        "F_dec": np.full((n, 128, 16, 16), np.nan, dtype=np.float32),
        "peak": np.full((n,), np.nan, dtype=np.float32),
    }
    tier: list[str] = [""] * n
    fps_try = [
        token_cache_id(cfg, dummy=False, stride=stride),
        token_cache_id(cfg, dummy=False),
        token_fingerprint(cfg, dummy=False),
    ]
    tok_caches = [TokenCache(cfg.path("paths.cache_root"), fp) for fp in fps_try]
    records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    sig = signature_from_cfg(cfg)
    frame_mem: dict[str, np.ndarray] = {}
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    warning = cfg.get("warning") or {}
    by_clip: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_clip.setdefault(row["clip_id"], []).append(i)
    filled_cache = 0
    filled_enc = 0
    encoder = None
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
                _write_pack(packs, tier, rows, warning, cache_rows[sl], out)
                filled_cache += len(cache_rows[sl])
        if need_encode and encode_missing:
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
                _write_pack(packs, tier, rows, warning, chunk, out)
                filled_enc += len(chunk)
        if ci % 32 == 0 or ci == len(by_clip):
            print(f"teacher_cache clips {ci}/{len(by_clip)}", flush=True)
    if encoder is not None:
        del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np.savez_compressed(
        out_dir / "teacher_pred.npz",
        H=packs["H"].astype(np.float16),
        Z=packs["Z"].astype(np.float16),
        F_dec=packs["F_dec"].astype(np.float16),
        peak=packs["peak"].astype(np.float16),
        row_id=np.arange(n, dtype=np.int32),
    )
    (out_dir / "tiers.jsonl").write_text("\n".join(json.dumps({"row_id": i, "tier": tier[i]}) for i in range(n)), encoding="utf-8")
    meta.update({
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "filled_token_cache": filled_cache,
        "filled_encode": filled_enc,
        "missing": int(n - filled_cache - filled_enc),
        "tensor_shapes": {"H": [5, 5], "Z": [32, 5, 5], "F_dec": [128, 16, 16]},
    })
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def subset_rows(
    full_rows: list[dict],
    full_gt: dict[str, np.ndarray],
    global_row_ids: list[int],
    *,
    split_label: str = "train_pilot",
) -> tuple[list[dict], dict[str, np.ndarray], list[int]]:
    """Compact a subset of full dense rows into a pilot cache (row_id 0..n-1)."""
    gids = [int(i) for i in global_row_ids]
    rows: list[dict] = []
    for new_id, gid in enumerate(gids):
        row = dict(full_rows[gid])
        row["row_id"] = new_id
        row["global_row_id"] = gid
        row["split"] = split_label
        rows.append(row)
    gt = {k: np.asarray(v[gids], dtype=np.float16) for k, v in full_gt.items()}
    return rows, gt, gids


@torch.no_grad()
def fill_teacher_cache_rows(
    cfg,
    out_dir: Path,
    *,
    rows: list[dict],
    gt: dict[str, np.ndarray],
    ckpt: Path,
    device: str,
    batch: int = 16,
    stride: int = 6,
    encode_missing: bool = True,
    meta_extra: dict | None = None,
) -> dict[str, Any]:
    """Encode teacher targets for an explicit row list (pilot subset)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"
    with index_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    np.savez_compressed(out_dir / "gt_heatmaps.npz", **gt)
    model, blob = load_pair_cf_parent(cfg, ckpt, device=device)
    n = len(rows)
    packs = {
        "H": np.full((n, 5, 5), np.nan, dtype=np.float32),
        "Z": np.full((n, 32, 5, 5), np.nan, dtype=np.float32),
        "F_dec": np.full((n, 128, 16, 16), np.nan, dtype=np.float32),
        "peak": np.full((n,), np.nan, dtype=np.float32),
    }
    tier: list[str] = [""] * n
    fps_try = [
        token_cache_id(cfg, dummy=False, stride=stride),
        token_cache_id(cfg, dummy=False),
        token_fingerprint(cfg, dummy=False),
    ]
    tok_caches = [TokenCache(cfg.path("paths.cache_root"), fp) for fp in fps_try]
    records = {r.clip_id: r for r in load_manifest(cfg.path("paths.manifest")) if r.valid}
    sig = signature_from_cfg(cfg)
    frame_mem: dict[str, np.ndarray] = {}
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    warning = cfg.get("warning") or {}
    by_clip: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_clip.setdefault(row["clip_id"], []).append(i)
    filled_cache = 0
    filled_enc = 0
    encoder = None
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
                _write_pack(packs, tier, rows, warning, cache_rows[sl], out)
                filled_cache += len(cache_rows[sl])
        if need_encode and encode_missing:
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
                _write_pack(packs, tier, rows, warning, chunk, out)
                filled_enc += len(chunk)
        if ci % 16 == 0 or ci == len(by_clip):
            print(f"teacher_cache_pilot clips {ci}/{len(by_clip)}", flush=True)
    if encoder is not None:
        del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    np.savez_compressed(
        out_dir / "teacher_pred.npz",
        H=packs["H"].astype(np.float16),
        Z=packs["Z"].astype(np.float16),
        F_dec=packs["F_dec"].astype(np.float16),
        peak=packs["peak"].astype(np.float16),
        row_id=np.arange(n, dtype=np.int32),
    )
    (out_dir / "tiers.jsonl").write_text("\n".join(json.dumps({"row_id": i, "tier": tier[i]}) for i in range(n)), encoding="utf-8")
    meta = {
        "stride_frames": stride,
        "n_rows": n,
        "splits": ["train_pilot"],
        "anchors_per_clip": int(meta_extra.get("anchors_per_clip", 0)) if meta_extra else 0,
        "preprocess_parent": signature_from_cfg(cfg).as_dict(),
        "checkpoint": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "filled_token_cache": filled_cache,
        "filled_encode": filled_enc,
        "missing": int(n - filled_cache - filled_enc),
        "tensor_shapes": {"H": [5, 5], "Z": [32, 5, 5], "F_dec": [128, 16, 16]},
        "purpose": "student_pilot_train_only",
    }
    if meta_extra:
        meta.update(meta_extra)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _write_pack(packs, tier, rows, warning, row_ids, out) -> None:
    h = out["Hhat_1s"].float().cpu().numpy()
    z = out["Z_t"].float().cpu().numpy()
    f = out["F_pre_decoder"].float().cpu().numpy()
    for j, rid in enumerate(row_ids):
        packs["H"][rid] = h[j]
        packs["Z"][rid] = z[j]
        packs["F_dec"][rid] = f[j]
        peak = float(h[j].max())
        packs["peak"][rid] = peak
        tname, _, _ = classify_heatmap(h[j], warning)
        tier[rid] = tname


def cache_integrity(out_dir: Path) -> dict[str, Any]:
    meta = json.loads((out_dir / "meta.json").read_text())
    pred = np.load(out_dir / "teacher_pred.npz")
    n = int(meta["n_rows"])
    rep = {
        "n_rows": n,
        "missing_H": int(np.isnan(pred["H"]).all(axis=(1, 2)).sum()),
        "H_finite_mean": float(np.nanmean(pred["H"])),
        "Z_finite_mean": float(np.nanmean(pred["Z"])),
        "F_dec_finite_mean": float(np.nanmean(pred["F_dec"])),
        "ordering": "index.jsonl row_id aligns with teacher_pred row_id",
        "meta": meta,
    }
    return rep
