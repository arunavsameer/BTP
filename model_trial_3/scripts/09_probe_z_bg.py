#!/usr/bin/env python3
"""Probe whether L_bg is trivial: compare Z_t cosine on bg-pairs vs unrelated clips.

Uses the same per-cell cosine as background_consistency_loss (mean_ij cos of 32-D cells).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg, resolve_checkpoint  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.pairing import background_pair_valid  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.models.teacher import CollisionTeacher3  # noqa: E402
from collision_jepa3.training.checkpoint import load_checkpoint, load_teacher_state  # noqa: E402
from collision_jepa3.utils.device import resolve_device  # noqa: E402


def cell_cosine(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """z: [32,5,5] → mean over 5×5 of cosine on the 32-D channel vector."""
    a = z_a.astype(np.float64).transpose(1, 2, 0).reshape(-1, z_a.shape[0])
    b = z_b.astype(np.float64).transpose(1, 2, 0).reshape(-1, z_b.shape[0])
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return float((a * b).sum(axis=1).mean())


def flat_cosine(z_a: np.ndarray, z_b: np.ndarray) -> float:
    a = z_a.astype(np.float64).ravel()
    b = z_b.astype(np.float64).ravel()
    a = a / max(np.linalg.norm(a), 1e-8)
    b = b / max(np.linalg.norm(b), 1e-8)
    return float(np.dot(a, b))


def summarize(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
        "mean_1_minus_cos": float(1.0 - arr.mean()),
    }


@torch.no_grad()
def encode_index(model, ds, rows: list[int], device: str, batch_size: int) -> dict[tuple[str, int], np.ndarray]:
    out: dict[tuple[str, int], np.ndarray] = {}
    amp = device.startswith("cuda")
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        samples = [ds[i] for i in chunk]
        frames = torch.from_numpy(np.stack([s["frames"] for s in samples], axis=0)).to(device)
        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z = model.encode(clip=frames)
        else:
            z = model.encode(clip=frames)
        z = z.float().cpu().numpy()
        for j, s in enumerate(samples):
            out[(s["clip_id"], int(s["anchor"]))] = z[j]
    return out


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-clips", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--out", default=None, help="JSON report path")
    args = parser.parse_args()
    cfg = load_cfg(args)
    device = resolve_device(args.device)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    ds = TeacherDataset(cfg, records, args.split, splits, use_token_cache=False)

    anchors_wanted = [90, 180, 270, 360, 405]
    anchors_wanted = [t for t in anchors_wanted if t in set(ds.anchors)]
    clip_ids = ds.clip_ids()
    rng = np.random.RandomState(int(cfg.get("project.seed", 1337)))
    if len(clip_ids) > args.n_clips:
        clip_ids = sorted(rng.choice(np.array(clip_ids, dtype=object), size=args.n_clips, replace=False).tolist())
    keep = set(clip_ids)
    rows = [i for i, (cid, t) in enumerate(ds.index) if cid in keep and int(t) in anchors_wanted]
    print(f"encoding {len(rows)} windows on {len(keep)} clips, anchors={anchors_wanted}", flush=True)

    model = CollisionTeacher3(cfg, dummy_backbone=bool(args.dummy))
    ckpt = resolve_checkpoint(cfg, args.checkpoint)
    blob = load_checkpoint(ckpt, map_location=device)
    load_teacher_state(model, blob)
    model.to(device).eval()
    zs = encode_index(model, ds, rows, device, args.batch_size)

    by_clip_t = zs
    recs = ds.records_by_id
    fork_default = int(cfg.get("sampling.fork_frame", 365))

    bg_pair = []
    unrelated = []
    same_clip_dt = []
    outcome_pre = []
    outcome_post = []

    # background pairs: same background_pair_id, same t
    by_pair: dict[tuple[str, int], list[str]] = defaultdict(list)
    for (cid, t) in by_clip_t:
        rec = recs[cid]
        pid = rec.background_pair_id or ""
        if pid:
            by_pair[(pid, t)].append(cid)
    for (_pid, t), cids in by_pair.items():
        if len(cids) < 2:
            continue
        a, b = cids[0], cids[1]
        rec_a, rec_b = recs[a], recs[b]
        ok = background_pair_valid(
            rec_a,
            rec_b,
            ds._heatmaps[a],
            ds._heatmaps[b],
            int(t),
            ds.horizons,
            ds.bg_tol,
        )
        if not ok:
            continue
        bg_pair.append(cell_cosine(by_clip_t[(a, t)], by_clip_t[(b, t)]))

    keys = list(by_clip_t.keys())
    # unrelated: different scenario_group, same t
    by_t: dict[int, list[str]] = defaultdict(list)
    for cid, t in keys:
        by_t[int(t)].append(cid)
    for t, cids in by_t.items():
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                ra, rb = recs[cids[i]], recs[cids[j]]
                if ra.scenario_group_id == rb.scenario_group_id:
                    continue
                if ra.background_pair_id and ra.background_pair_id == rb.background_pair_id:
                    continue
                unrelated.append(cell_cosine(by_clip_t[(cids[i], t)], by_clip_t[(cids[j], t)]))

    # same clip, different times
    by_clip: dict[str, list[int]] = defaultdict(list)
    for cid, t in keys:
        by_clip[cid].append(int(t))
    for cid, ts in by_clip.items():
        ts = sorted(set(ts))
        for i in range(len(ts) - 1):
            same_clip_dt.append(cell_cosine(by_clip_t[(cid, ts[i])], by_clip_t[(cid, ts[i + 1])]))

    # collision vs miss, same sample + bg
    by_out: dict[tuple[str, str, int], dict[str, str]] = defaultdict(dict)
    for cid, t in keys:
        rec = recs[cid]
        by_out[(rec.scenario_group_id, rec.background_id, int(t))][rec.outcome_type] = cid
    for (sample, bg, t), outs in by_out.items():
        if "collision" not in outs:
            continue
        miss = next((k for k in outs if k.startswith("miss")), None)
        if miss is None:
            continue
        c_id, m_id = outs["collision"], outs[miss]
        cos = cell_cosine(by_clip_t[(c_id, t)], by_clip_t[(m_id, t)])
        fork = int(recs[c_id].fork_frame or fork_default)
        if t < fork:
            outcome_pre.append(cos)
        else:
            outcome_post.append(cos)

    # subsample unrelated if huge
    if len(unrelated) > 4000:
        unrelated = rng.choice(np.asarray(unrelated), size=4000, replace=False).tolist()

    report = {
        "checkpoint": str(ckpt),
        "split": args.split,
        "n_windows": len(by_clip_t),
        "n_clips": len(keep),
        "anchors": anchors_wanted,
        "cell_cosine": {
            "bg_pair_valid": summarize(bg_pair),
            "unrelated_same_t": summarize(unrelated),
            "same_clip_adjacent_t": summarize(same_clip_dt),
            "collision_vs_miss_pre_fork": summarize(outcome_pre),
            "collision_vs_miss_post_fork": summarize(outcome_post),
        },
    }
    out = Path(args.out) if args.out else cfg.path("paths.reports_root") / "z_bg_probe.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
