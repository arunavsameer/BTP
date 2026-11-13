#!/usr/bin/env python3
"""CaDeT-plan diagnostics on frozen G. Does not train or overwrite G.

1) Dump-only pair oracles (average / better / worse of bg mates).
2) Linear probes: can F / Z_t / threat-cells / bg-cells classify bg_id and lighting?
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import warning_report  # noqa: E402
from collision_jepa3.training.engine import move_batch  # noqa: E402
from collision_jepa3.u1_runtime import load_dump, load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402

CAUTION, STOP = 0.25, 0.60
LABELS = ("SAFE", "THREAT", "STOP")


def _warn(peak: float) -> str:
    if peak >= STOP:
        return "STOP"
    if peak >= CAUTION:
        return "THREAT"
    return "SAFE"


def _pair_key(cid: str) -> tuple[str, str]:
    parts = str(cid).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}|{parts[1]}", parts[2]
    return str(cid), "?"


def _headline(rep: dict) -> dict:
    h = dict(rep["headline"])
    h["mse"] = float(rep["all"]["operating"]["mse"])
    return h


def dump_pair_oracles(dump: dict) -> dict:
    a = dump["arrays"]
    m = dump["meta"]
    pred = np.asarray(a["pred_mu"], dtype=np.float32)
    gt = np.asarray(a["gt_clip"], dtype=np.float32)
    pre = np.asarray(a["pre_fork"], dtype=bool)
    coll = np.asarray(a["collision"], dtype=bool)
    blob = np.asarray(a["blob_n"])
    paired = np.asarray(a["paired"], dtype=bool)
    anchor = np.asarray(a["anchor"])
    clip_id = np.asarray(m["clip_id"])
    n = len(pred)
    buckets: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for i, cid in enumerate(clip_id):
        pid, bg = _pair_key(cid)
        buckets[(pid, int(anchor[i]))][bg] = i
    mate = np.full(n, -1, dtype=np.int32)
    complete = 0
    for _k, bgs in buckets.items():
        if "bg_00" in bgs and "bg_01" in bgs:
            i0, i1 = bgs["bg_00"], bgs["bg_01"]
            mate[i0] = i1
            mate[i1] = i0
            complete += 1
    avg = pred.copy()
    better = pred.copy()
    worse = pred.copy()
    mae = np.mean(np.abs(pred - gt), axis=(1, 2))
    flip_rows = []
    for i in range(n):
        j = int(mate[i])
        if j < 0:
            continue
        avg[i] = 0.5 * (pred[i] + pred[j])
        if mae[j] < mae[i]:
            better[i] = pred[j]
            worse[i] = pred[i]
        else:
            better[i] = pred[i]
            worse[i] = pred[j]
        p0, p1 = float(pred[i].max()), float(pred[j].max())
        w0, w1, wg = _warn(p0), _warn(p1), _warn(float(gt[i].max()))
        if i < j and w0 != w1:
            flip_rows.append(
                {
                    "pre": bool(pre[i]),
                    "w0": w0,
                    "w1": w1,
                    "gt": wg,
                    "correct0": w0 == wg,
                    "correct1": w1 == wg,
                    "avg_w": _warn(float(avg[i].max())),
                }
            )

    def score(name, arr):
        return {
            "name": name,
            "headline": _headline(
                warning_report(arr, gt, pre_fork=pre, collision=coll, blob_n=blob, paired=paired)
            )
        }

    n_flip = len(flip_rows)
    both_wrong = sum(1 for r in flip_rows if not r["correct0"] and not r["correct1"])
    one_right = sum(1 for r in flip_rows if r["correct0"] != r["correct1"])
    avg_fixes = sum(1 for r in flip_rows if r["avg_w"] == r["gt"] and r["correct0"] != r["correct1"])
    avg_breaks = sum(
        1
        for r in flip_rows
        if r["avg_w"] != r["gt"] and (r["correct0"] or r["correct1"])
    )
    return {
        "n_windows": n,
        "n_complete_pairs": complete,
        "oracles": [
            score("G", pred),
            score("pair_mean (E1 oracle)", avg),
            score("pair_better (not deployable)", better),
            score("pair_worse", worse),
        ],
        "flips": {
            "n_pairs": complete,
            "n_warn_flip_pairs": n_flip,
            "flip_rate": n_flip / max(complete, 1),
            "among_flips_both_wrong": both_wrong,
            "among_flips_exactly_one_correct": one_right,
            "avg_recovers_the_correct_class": avg_fixes,
            "avg_destroys_a_correct_class": avg_breaks,
            "pre_fork_flips": sum(1 for r in flip_rows if r["pre"]),
            "gt_class_of_flips": {
                k: sum(1 for r in flip_rows if r["gt"] == k) for k in LABELS
            },
        },
    }


@torch.no_grad()
def collect_pools(model, dataset, cfg, device: str, records_by_id: dict) -> dict:
    cfg.set("training.num_workers", 0)
    loader = build_dataloader(dataset, cfg, shuffle=False)
    model.eval()
    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        z, fine = model.encode_pair(tokens=batch["tokens"])
        z = z.float()
        fine = fine.float()
        gt = batch["h_1s"].float()
        w = F.interpolate(gt.unsqueeze(1), size=fine.shape[-2:], mode="bilinear", align_corners=False)
        w = w.clamp(0, 1)
        bgw = (1.0 - w).clamp(min=0)
        # If a window is empty, fall back to uniform so the pool is defined.
        tsum = w.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6)
        bsum = bgw.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-6)
        f_mean = fine.mean(dim=(2, 3))
        z_mean = z.mean(dim=(2, 3))
        z_flat = z.flatten(1)
        f_threat = (fine * w).sum(dim=(2, 3)) / tsum
        f_bg = (fine * bgw).sum(dim=(2, 3)) / bsum
        ids = list(batch["clip_id"])
        anc = batch["anchor"].detach().cpu().numpy().astype(np.int64)
        for i, cid in enumerate(ids):
            rec = records_by_id[cid]
            pid, bg = _pair_key(cid)
            rows.append(
                {
                    "clip_id": cid,
                    "pair_id": pid,
                    "bg": bg,
                    "lighting": rec.lighting or "unknown",
                    "group": rec.scenario_group_id,
                    "anchor": int(anc[i]),
                    "f_mean": f_mean[i].cpu().numpy(),
                    "f_threat": f_threat[i].cpu().numpy(),
                    "f_bg": f_bg[i].cpu().numpy(),
                    "z_mean": z_mean[i].cpu().numpy(),
                    "z_flat": z_flat[i].cpu().numpy(),
                }
            )
        if len(rows) % 640 == 0 or len(rows) == len(dataset):
            print(f"  pooled {len(rows)}/{len(dataset)}", flush=True)
    return rows


def _probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, name: str) -> dict:
    labels, y_idx = np.unique(y, return_inverse=True)
    n_cls = len(labels)
    chance = 1.0 / n_cls
    n_groups = len(np.unique(groups))
    splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=splits)
    accs, baccs = [], []
    for tr, te in gkf.split(X, y_idx, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=400, class_weight="balanced"),
        )
        clf.fit(X[tr], y_idx[tr])
        pred = clf.predict(X[te])
        accs.append(float(accuracy_score(y_idx[te], pred)))
        baccs.append(float(balanced_accuracy_score(y_idx[te], pred)))
    return {
        "name": name,
        "n": int(len(y)),
        "n_classes": int(n_cls),
        "classes": [str(x) for x in labels],
        "chance": float(chance),
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "balanced_acc_mean": float(np.mean(baccs)),
        "delta_vs_chance": float(np.mean(baccs) - chance),
    }


def run_probes(rows: list[dict]) -> dict:
    groups = np.array([r["group"] for r in rows])
    bg = np.array([r["bg"] for r in rows])
    lighting = np.array([r["lighting"] for r in rows])
    feats = {
        "F global pool": np.stack([r["f_mean"] for r in rows]),
        "F threat-weighted (GT H>0)": np.stack([r["f_threat"] for r in rows]),
        "F background-weighted (1-H)": np.stack([r["f_bg"] for r in rows]),
        "Z_t global pool": np.stack([r["z_mean"] for r in rows]),
        "Z_t flattened 32x5x5": np.stack([r["z_flat"] for r in rows]),
    }
    out = {"bg_id": [], "lighting": []}
    for name, X in feats.items():
        out["bg_id"].append(_probe(X, bg, groups, name))
        out["lighting"].append(_probe(X, lighting, groups, name))
    # Pair residual energy: |F00-F01| on threat vs bg pools (cosine of pooled vectors).
    by = defaultdict(dict)
    for i, r in enumerate(rows):
        by[(r["pair_id"], r["anchor"])][r["bg"]] = i
    threat_cos, bg_cos, mean_cos, z_cos = [], [], [], []
    for _k, m in by.items():
        if "bg_00" not in m or "bg_01" not in m:
            continue
        i0, i1 = m["bg_00"], m["bg_01"]

        def cos(a, b):
            a = a.astype(np.float64)
            b = b.astype(np.float64)
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-8 or nb < 1e-8:
                return 1.0 if na < 1e-8 and nb < 1e-8 else 0.0
            return float(np.dot(a, b) / (na * nb))

        threat_cos.append(cos(rows[i0]["f_threat"], rows[i1]["f_threat"]))
        bg_cos.append(cos(rows[i0]["f_bg"], rows[i1]["f_bg"]))
        mean_cos.append(cos(rows[i0]["f_mean"], rows[i1]["f_mean"]))
        z_cos.append(cos(rows[i0]["z_flat"], rows[i1]["z_flat"]))
    out["pair_cosine"] = {
        "F_threat": {"mean": float(np.mean(threat_cos)), "p10": float(np.quantile(threat_cos, 0.1))},
        "F_bg": {"mean": float(np.mean(bg_cos)), "p10": float(np.quantile(bg_cos, 0.1))},
        "F_mean": {"mean": float(np.mean(mean_cos)), "p10": float(np.quantile(mean_cos, 0.1))},
        "Z_flat": {"mean": float(np.mean(z_cos)), "p10": float(np.quantile(z_cos, 0.1))},
        "n_pairs": len(threat_cos),
    }
    return out


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--dump", default="reports/direct_h1s_u1/dump_val.npz")
    parser.add_argument("--out", default="reports/cadet_diag/diagnostics.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", False)
    cfg.set("training.num_workers", 0)
    device = resolve_device(args.device)
    out = {"hardware": describe_hardware(device), "dump_oracles": None, "probes": None}
    print(json.dumps(out["hardware"], indent=2), flush=True)

    dump = load_dump(Path(args.dump))
    out["dump_oracles"] = dump_pair_oracles(dump)
    print(json.dumps(out["dump_oracles"], indent=2, default=str), flush=True)

    if not args.skip_probe:
        records = load_manifest(cfg.path("paths.manifest"))
        splits = load_splits(cfg.path("paths.split_file"))
        recs = {r.clip_id: r for r in records}
        ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
        model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=False)
        print("extracting F/Z pools on val", flush=True)
        rows = collect_pools(model, ds, cfg, device, recs)
        out["probes"] = run_probes(rows)
        print(json.dumps(out["probes"], indent=2), flush=True)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
