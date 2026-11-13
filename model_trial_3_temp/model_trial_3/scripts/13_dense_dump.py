#!/usr/bin/env python3
"""Dense val dump: one row per clip×anchor (default 0.2 s) with GT maps and model H/Z.

Layout (aligned by row order):
  reports/dense_dump/meta.json
  reports/dense_dump/index.jsonl          scalar fields, one object per row
  reports/dense_dump/gt_heatmaps.npz      gt_now, gt_1s, gt_1_5s, gt_2s [N,5,5] float16
  reports/dense_dump/pred_<name>.npz      H [N,5,5], Z [N,32,5,5] float16; missing rows are NaN
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.cache import TokenCache, load_cached_frames, token_fingerprint  # noqa: E402
from collision_jepa3.data.heatmap_annotations import load_heatmaps, targets_for_anchor  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.pairing import compute_ambiguity_table  # noqa: E402
from collision_jepa3.data.preprocessing import uint8_to_nchw01  # noqa: E402
from collision_jepa3.data.sampling import FRAME_OFFSETS, HORIZONS_FRAMES, input_indices, iter_anchors  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.labels import row_scalars, true_side_for_group  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.training.engine import teacher_forward  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


INDEX_FIELDS = [
    "row_id", "clip_id", "sample_id", "frame", "fps",
    "outcome", "outcome_raw", "miss_side", "fork_frame", "collision_frame", "hit_s",
    "scenario_type", "true_side", "background_id", "background_pair_id", "outcome_group_id",
    "pre_fork", "amb_1s", "amb_1_5s", "amb_2s", "tti_s", "anchor_class",
    "collision_within_1s", "collision_within_2s", "collision_within_3s", "collision_within_4s",
]


def _parse_spec(item: str) -> tuple[str, Path, str]:
    raw = item
    mode = "direct_h1s"
    if raw.count(":") >= 1 and raw.rsplit(":", 1)[-1] in {"direct_h1s", "residual"}:
        raw, mode = raw.rsplit(":", 1)
    if "=" in raw:
        name, path = raw.split("=", 1)
    else:
        path = raw
        name = Path(path).parent.name
    return name, Path(path), mode


def _dummy() -> DummyVJEPA2Backbone:
    return DummyVJEPA2Backbone(
        hidden_size=1408, crop_size=256, patch_size=16, tubelet_size=2,
        num_frames=16, num_layers=1,
    )


def _load_heads(cfg, path: Path, mode: str, device: str):
    cfg.set("heads.mode", mode)
    model = CollisionTeacher3(cfg, backbone=_dummy())
    blob = load_checkpoint(str(path), map_location="cpu")
    load_teacher_state(model, blob)
    model.to(device).eval()
    return model, blob.get("epoch")


def build_index(cfg, split: str, stride: int) -> tuple[list[dict], dict[str, np.ndarray], dict]:
    records = [r for r in load_manifest(cfg.path("paths.manifest")) if r.valid]
    splits = load_splits(cfg.path("paths.split_file"))
    groups = set(splits.get(split, []))
    recs = [r for r in records if r.scenario_group_id in groups]
    n_frames = int(cfg.get("data.expected_frames", 450))
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    horizons = list(cfg.get("sampling.horizons_frames", HORIZONS_FRAMES))
    names = list(cfg.get("sampling.horizon_names", ["1s", "1_5s", "2s"]))
    anchors = iter_anchors(
        n_frames, int(stride), offsets, horizons,
        allow_partial_horizons=bool(cfg.get("sampling.allow_partial_horizons", True)),
    )
    heatmaps = {
        r.clip_id: load_heatmaps(r.heatmap_json_path, expected_frames=n_frames, grid=5)
        for r in recs
    }
    amb = compute_ambiguity_table(
        heatmaps, recs,
        fork_frame=int(cfg.get("sampling.fork_frame", 365)),
        n_frames=n_frames, stride=int(stride), offsets=offsets, horizons=horizons,
        horizon_names=names,
        heatmap_tol=float(cfg.get("sampling.heatmap_disagree_tol", 0.02)),
        mask_ambiguous=True,
        allow_partial_horizons=True,
    )
    collision_of = {}
    for r in recs:
        if r.outcome_type == "collision" and r.outcome_group_id:
            collision_of[r.outcome_group_id] = r.clip_id
    true_side = {}
    warning = cfg.get("warning")
    for gid, cid in collision_of.items():
        rec = next(x for x in recs if x.clip_id == cid)
        true_side[gid] = true_side_for_group(heatmaps[cid], rec.scenario_type, warning)

    rows = []
    gt = {k: [] for k in ("gt_now", "gt_1s", "gt_1_5s", "gt_2s")}
    fps_vals = []
    for rec in recs:
        fps = float(rec.fps or cfg.get("data.fps_expected", 30))
        fps_vals.append(fps)
        side = true_side.get(rec.outcome_group_id or "", side_from_rec(rec, heatmaps, warning))
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
            rows.append(scalar)
            gt["gt_now"].append(tg["h_now"])
            gt["gt_1s"].append(tg["h_1s"])
            gt["gt_1_5s"].append(tg["h_1_5s"])
            gt["gt_2s"].append(tg["h_2s"])
    stacked = {k: np.stack(v, axis=0).astype(np.float16) for k, v in gt.items()}
    meta = {
        "split": split,
        "n_clips": len(recs),
        "n_rows": len(rows),
        "n_anchors_per_clip": len(anchors),
        "anchors": anchors,
        "stride_frames": int(stride),
        "stride_s": float(stride) / float(np.median(fps_vals) if fps_vals else 30),
        "fps_unique": sorted(set(float(x) for x in fps_vals)),
        "scenario_types": sorted({r.scenario_type for r in recs}),
        "index_fields": INDEX_FIELDS,
        "gt_npz_keys": list(stacked.keys()),
        "pred_npz_keys": ["H", "Z"],
        "z_shape": [32, 5, 5],
        "heatmap_shape": [5, 5],
        "note": "RAW GT heatmaps (not mean-target training labels). amb_* marks pre-fork horizons that disagree across collision/miss mates.",
    }
    return rows, stacked, meta


def side_from_rec(rec, heatmaps, warning) -> str:
    from collision_jepa3.evaluation.labels import side_from_heatmap, side_from_scenario

    named = side_from_scenario(rec.scenario_type)
    if named:
        return named
    hm = heatmaps[rec.clip_id]
    peaks = hm.reshape(len(hm), -1).max(axis=1)
    return side_from_heatmap(hm[int(np.argmax(peaks))], warning)


def write_dump(out_dir: Path, rows: list[dict], gt: dict[str, np.ndarray], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    with (out_dir / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    np.savez_compressed(out_dir / "gt_heatmaps.npz", **gt)
    print(f"wrote {len(rows)} rows → {out_dir}", flush=True)


def _infer_on_tokens(model, tokens: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        out = teacher_forward(model, {"tokens": tokens})
    h = out["Hhat_1s"].float().cpu().numpy()
    z = out["Z_t"].float().cpu().numpy()
    return h, z


def fill_models(
    cfg, rows: list[dict], specs: list[tuple[str, Path, str]], device: str,
    out_dir: Path, encode_missing: bool, batch: int,
) -> dict:
    """Encode each clip once, then run every head on those tokens."""
    n = len(rows)
    tok_cache = TokenCache(cfg.path("paths.cache_root"), token_fingerprint(cfg, dummy=False))
    models = {}
    packs = {}
    counts = {}
    for name, path, mode in specs:
        model, epoch = _load_heads(cfg, path, mode, device)
        models[name] = model
        packs[name] = {
            "H": np.full((n, 5, 5), np.nan, dtype=np.float32),
            "Z": np.full((n, 32, 5, 5), np.nan, dtype=np.float32),
        }
        counts[name] = {"checkpoint": str(path), "heads_mode": mode, "epoch": epoch, "filled_from_token_cache": 0, "filled_from_encode": 0}

    def _run(tokens: torch.Tensor, rids: list[int], source: str) -> None:
        for name, model in models.items():
            h, z = _infer_on_tokens(model, tokens)
            for j, rid in enumerate(rids):
                packs[name]["H"][rid] = h[j]
                packs[name]["Z"][rid] = z[j]
            counts[name][f"filled_from_{source}"] += len(rids)

    encoder = None
    by_clip: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_clip.setdefault(row["clip_id"], []).append(i)
    offsets = list(cfg.get("sampling.frame_offsets", FRAME_OFFSETS))
    n_clips = len(by_clip)
    for ci, (cid, idxs) in enumerate(by_clip.items(), start=1):
        mapping = {}
        arr = None
        try:
            arr, mapping = tok_cache.load_clip(cid, mmap=True)
        except FileNotFoundError:
            mapping = {}
        cache_rows, cache_toks, need_encode = [], [], []
        for i in idxs:
            t = int(rows[i]["frame"])
            if t in mapping:
                cache_rows.append(i)
                cache_toks.append(np.asarray(arr[mapping[t]], dtype=np.float32))
            else:
                need_encode.append(i)
        if cache_toks:
            x = torch.from_numpy(np.stack(cache_toks, axis=0)).to(device)
            for start in range(0, len(cache_rows), batch):
                sl = slice(start, start + batch)
                _run(x[sl], cache_rows[sl], "token_cache")
        if need_encode and encode_missing:
            if encoder is None:
                print("loading Giant encoder for missing anchors", flush=True)
                encoder = CollisionTeacher3(cfg, dummy_backbone=False).to(device).eval()
            frames = load_cached_frames(cfg.path("paths.cache_root"), cid, mmap=True)
            for start in range(0, len(need_encode), batch):
                chunk = need_encode[start : start + batch]
                windows = [
                    uint8_to_nchw01(np.stack([np.asarray(frames[k]) for k in input_indices(int(rows[rid]["frame"]), offsets)], axis=0))
                    for rid in chunk
                ]
                clips = torch.from_numpy(np.stack(windows, axis=0)).to(device)
                with torch.no_grad():
                    if str(device).startswith("cuda"):
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            tokens = encoder.encode_tokens(clips)
                    else:
                        tokens = encoder.encode_tokens(clips)
                _run(tokens, chunk, "encode")
        if ci % 16 == 0 or ci == n_clips:
            print(f"  clips {ci}/{n_clips}", flush=True)

    if encoder is not None:
        del encoder
    for name, model in models.items():
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for name, pack in packs.items():
        np.savez_compressed(
            out_dir / f"pred_{name}.npz",
            H=pack["H"].astype(np.float16),
            Z=pack["Z"].astype(np.float16),
            row_id=np.arange(n, dtype=np.int32),
        )
        filled = counts[name]["filled_from_token_cache"] + counts[name]["filled_from_encode"]
        counts[name]["missing"] = int(n - filled)
        print(json.dumps({name: counts[name]}), flush=True)
    return counts


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--split", default="val")
    parser.add_argument("--stride", type=int, default=6, help="anchor stride in frames; 6 ≈ 0.2 s at 30 fps")
    parser.add_argument("--out", default=None)
    parser.add_argument("--gt-only", action="store_true")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--encode-missing", action="store_true", help="run Giant on anchors not in the token cache")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    cfg = load_cfg(args)
    out_dir = Path(args.out) if args.out else cfg.path("paths.reports_root") / "dense_dump"
    index_path = out_dir / "index.jsonl"
    if index_path.exists() and (out_dir / "gt_heatmaps.npz").exists():
        print("reusing GT dump", out_dir, flush=True)
        rows = [json.loads(l) for l in index_path.open()]
        meta = json.loads((out_dir / "meta.json").read_text())
    else:
        rows, gt, meta = build_index(cfg, args.split, args.stride)
        write_dump(out_dir, rows, gt, meta)

    if args.gt_only:
        return
    device = resolve_device(args.device)
    specs = args.models or [
        "phase_a=checkpoints/teacher_phase_a_best.pt:residual",
        "mse17=checkpoints/direct_h1s_mse_until/teacher_phase_a_best.pt:direct_h1s",
        "beta2=checkpoints/direct_h1s_beta2/teacher_phase_a_best.pt:direct_h1s",
    ]
    parsed = [_parse_spec(s) for s in specs]
    preds = fill_models(cfg, rows, parsed, device, out_dir, args.encode_missing, args.batch)
    meta = json.loads((out_dir / "meta.json").read_text())
    meta["models"] = preds
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
