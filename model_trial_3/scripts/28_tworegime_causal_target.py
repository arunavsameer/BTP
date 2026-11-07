#!/usr/bin/env python3
"""Two-regime causal target: dropout MLE when y is STOP, quiet MLE when y is SAFE.

Does not train or overwrite G.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import warning_report  # noqa: E402
from collision_jepa3.u1_runtime import load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402

STOP, CAUTION = 0.60, 0.25


def _load24():
    p = Path(__file__).with_name("24_cadet_oracles.py")
    spec = importlib.util.spec_from_file_location("cadet24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pair_lo_hi(mate: np.ndarray):
    i = np.arange(len(mate))
    ok = (mate >= 0) & (i < mate)
    return i[ok], mate[ok]


def mix_pairs(fine, lo, hi, w_lo):
    lo_t, hi_t = torch.from_numpy(lo), torch.from_numpy(hi)
    w = w_lo.view(-1, 1, 1, 1)
    fc = w * fine[lo_t] + (1.0 - w) * fine[hi_t]
    out = fine.clone()
    out[lo_t] = fc
    out[hi_t] = fc
    return out


def softmax2(a, b, temp):
    x = np.stack([a, b], axis=1) / max(temp, 1e-4)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e[:, 0] / e.sum(axis=1)).astype(np.float32)


def blob_energy(pred, gt):
    """Mean predicted mass on GT-occupied cells; peak if the map is empty."""
    occ = gt >= 0.5
    out = pred.reshape(len(pred), -1).max(1).astype(np.float64)
    p = pred.reshape(len(pred), -1)
    g = occ.reshape(len(gt), -1)
    n = g.sum(axis=1)
    has = n > 0
    out[has] = (p * g).sum(axis=1)[has] / np.clip(n[has], 1, None)
    return out


def two_regime_w(gt_peak, peaks, lo, hi, *, blob=None, soft=False, temp=0.15):
    """STOP: mass on the view that still encodes the object. SAFE: mass on the quiet view."""
    n = len(lo)
    w = np.full(n, 0.5, dtype=np.float32)
    stop = (gt_peak[lo] >= STOP) | (gt_peak[hi] >= STOP)
    safe = (gt_peak[lo] < CAUTION) & (gt_peak[hi] < CAUTION)
    score = blob if blob is not None else peaks
    if soft:
        w[stop] = softmax2(score[lo[stop]], score[hi[stop]], temp)
        w[safe] = softmax2(-peaks[lo[safe]], -peaks[hi[safe]], temp)
    else:
        w[stop] = (score[lo[stop]] >= score[hi[stop]]).astype(np.float32)
        w[safe] = (peaks[lo[safe]] <= peaks[hi[safe]]).astype(np.float32)
    return torch.from_numpy(w)


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/tworegime_causal.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    c24 = _load24()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", False)
    cfg.set("training.num_workers", 0)
    device = resolve_device(args.device)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    recs = {r.clip_id: r for r in records}
    ds = TeacherDataset(cfg, records, "val", splits, dummy_backbone=False, use_token_cache=True)
    model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=False)
    print("collect F", flush=True)
    pack = c24.collect(model, ds, cfg, device, recs)
    fine, mate = pack["fine"], pack["mate"]
    lo, hi = pair_lo_hi(mate)
    pred_g = c24.decode(model, fine, pack["z"], device)
    g_rep = warning_report(
        pred_g, pack["gt"].numpy(), pre_fork=pack["pre"], collision=pack["coll"],
        blob_n=pack["blob"], paired=pack["paired"],
    )
    gt = pack["gt"].numpy()
    peaks = pred_g.reshape(len(pred_g), -1).max(1)
    gt_peak = gt.reshape(len(gt), -1).max(1)
    blob = blob_energy(pred_g, gt)
    mse = ((pred_g.reshape(len(pred_g), -1) - gt.reshape(len(gt), -1)) ** 2).mean(1)

    tau = 1.0 / np.clip(mse, 1e-8, None)
    w_gm = (tau[lo] / (tau[lo] + tau[hi])).astype(np.float32)

    plans = {
        "G": None,
        "pair-mean": torch.full((len(lo),), 0.5),
        "peak-softmax T=0.15 (always loud)": torch.from_numpy(softmax2(peaks[lo], peaks[hi], 0.15)),
        "inverse-MSE (always quiet-ish)": torch.from_numpy(w_gm),
        "two-regime hard (peak vis / quiet)": two_regime_w(gt_peak, peaks, lo, hi, soft=False),
        "two-regime soft T=0.15": two_regime_w(gt_peak, peaks, lo, hi, soft=True, temp=0.15),
        "two-regime hard (blob vis / quiet)": two_regime_w(gt_peak, peaks, lo, hi, blob=blob, soft=False),
        "two-regime soft blob T=0.15": two_regime_w(gt_peak, peaks, lo, hi, blob=blob, soft=True, temp=0.15),
    }
    results = []
    for name, w_lo in plans.items():
        print("decode", name, flush=True)
        if w_lo is None:
            pred = pred_g
            row = c24.score_plan(name, pred, pack, None)
        else:
            pred = c24.decode(model, mix_pairs(fine, lo, hi, w_lo), pack["z"], device)
            row = c24.score_plan(name, pred, pack, g_rep)
        results.append(row)
        h = row["headline"]
        feas = row.get("feasible_vs_g") or {}
        mark = " FEASIBLE" if feas.get("feasible") else (" no" if feas else "")
        print(
            f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} pre={h['pre_fork_stop_recall']:.3f} "
            f"rec5={h['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
            f"night={row['lighting_stop'].get('night', {}).get('stop_recall')} "
            f"glare={row['lighting_stop'].get('harsh_glare', {}).get('stop_recall')}{mark}",
            flush=True,
        )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"n": len(pred_g), "n_pairs": pack["n_pairs"], "results": results}, indent=2, default=str),
        encoding="utf-8",
    )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
