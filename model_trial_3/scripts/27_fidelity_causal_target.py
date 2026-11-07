#!/usr/bin/env python3
"""Causal F as a fidelity-weighted combination of two environment measurements.

Does not train or overwrite G. Tests a professor-defensible target:
each street is a noisy measurement of the same invariant latent PC;
combine them by how well a frozen readout recovers the invariant label y,
not by model confidence and not by an equal mean.
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

W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load24():
    p = Path(__file__).with_name("24_cadet_oracles.py")
    spec = importlib.util.spec_from_file_location("cadet24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def map_mse(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    p = pred.reshape(len(pred), -1).astype(np.float64)
    g = gt.reshape(len(gt), -1).astype(np.float64)
    return ((p - g) ** 2).mean(axis=1)


def pair_lo_hi(mate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i = np.arange(len(mate))
    ok = (mate >= 0) & (i < mate)
    return i[ok], mate[ok]


def same_on_pairs(fine: torch.Tensor, lo: np.ndarray, hi: np.ndarray, fc: torch.Tensor) -> torch.Tensor:
    out = fine.clone()
    out[torch.from_numpy(lo)] = fc
    out[torch.from_numpy(hi)] = fc
    return out


def mix_pairs(fine: torch.Tensor, lo: np.ndarray, hi: np.ndarray, w_lo: torch.Tensor) -> torch.Tensor:
    """F_c = w_lo F_lo + (1-w_lo) F_hi, copied to both members."""
    lo_t, hi_t = torch.from_numpy(lo), torch.from_numpy(hi)
    w = w_lo.view(-1, 1, 1, 1)
    fc = w * fine[lo_t] + (1.0 - w) * fine[hi_t]
    return same_on_pairs(fine, lo, hi, fc)


def inverse_mse_weights(mse: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> torch.Tensor:
    """Gauss-Markov: w_lo = τ_lo / (τ_lo+τ_hi), τ = 1/MSE. Returns w on lo."""
    tau_lo = 1.0 / np.clip(mse[lo], 1e-8, None)
    tau_hi = 1.0 / np.clip(mse[hi], 1e-8, None)
    return torch.from_numpy((tau_lo / (tau_lo + tau_hi)).astype(np.float32))


def better_view_w(mse: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> torch.Tensor:
    """Hard pick the lower-MSE view. Tie → 0.5."""
    w = np.full(len(lo), 0.5, dtype=np.float32)
    w[mse[lo] < mse[hi]] = 1.0
    w[mse[hi] < mse[lo]] = 0.0
    return torch.from_numpy(w)


def peak_softmax_w(peaks: np.ndarray, lo: np.ndarray, hi: np.ndarray, temp: float) -> torch.Tensor:
    a = np.stack([peaks[lo], peaks[hi]], axis=1) / max(temp, 1e-4)
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return torch.from_numpy((e[:, 0] / e.sum(axis=1)).astype(np.float32))


def line_search_w(model, fine, z, gt, lo, hi, device, c24) -> tuple[torch.Tensor, dict]:
    """Per pair, pick w in W_GRID minimizing MSE(D(w F_lo + (1-w) F_hi), y)."""
    n_p = len(lo)
    best = np.full(n_p, 0.5, dtype=np.float32)
    best_mse = np.full(n_p, np.inf, dtype=np.float64)
    hist = {str(w): 0 for w in W_GRID}
    for w in W_GRID:
        w_lo = torch.full((n_p,), float(w))
        f_star = mix_pairs(fine, lo, hi, w_lo)
        pred = c24.decode(model, f_star, z, device)
        m = 0.5 * (map_mse(pred, gt)[lo] + map_mse(pred, gt)[hi])
        take = m < best_mse
        best[take] = w
        best_mse[take] = m[take]
        hist[str(w)] = int(take.sum()) if w == W_GRID[0] else int((best == w).sum())
    hist = {str(w): int((best == w).sum()) for w in W_GRID}
    return torch.from_numpy(best), {
        "grid": list(W_GRID),
        "counts": hist,
        "frac_mean": hist["0.5"] / max(n_p, 1),
        "frac_vertex": (hist["0.0"] + hist["1.0"]) / max(n_p, 1),
        "mean_mse": float(best_mse.mean()),
    }


def weight_stats(name: str, w_lo: torch.Tensor, mse: np.ndarray, peaks: np.ndarray, gt_peak: np.ndarray, lo, hi) -> dict:
    w = w_lo.numpy()
    stop = (gt_peak[lo] >= 0.60) | (gt_peak[hi] >= 0.60)
    safe = (gt_peak[lo] < 0.25) & (gt_peak[hi] < 0.25)
    pick_lo = w >= 0.5
    # among SAFE pairs, did we put more mass on the *quieter* (lower peak) view?
    quieter_is_lo = peaks[lo] <= peaks[hi]
    better_mse_is_lo = mse[lo] <= mse[hi]
    louder_is_lo = peaks[lo] >= peaks[hi]
    return {
        "name": name,
        "mean_w_lo": float(w.mean()),
        "frac_near_mean": float(np.mean(np.abs(w - 0.5) < 0.1)),
        "frac_near_vertex": float(np.mean((w < 0.1) | (w > 0.9))),
        "safe_picks_quieter": float(np.mean((pick_lo == quieter_is_lo)[safe])) if safe.any() else None,
        "safe_picks_better_mse": float(np.mean((pick_lo == better_mse_is_lo)[safe])) if safe.any() else None,
        "stop_picks_louder": float(np.mean((pick_lo == louder_is_lo)[stop])) if stop.any() else None,
        "stop_picks_better_mse": float(np.mean((pick_lo == better_mse_is_lo)[stop])) if stop.any() else None,
        "n_safe_pairs": int(safe.sum()),
        "n_stop_pairs": int(stop.sum()),
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/fidelity_causal.json")
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
    fine, mate, z, gt = pack["fine"], pack["mate"], pack["z"], pack["gt"]
    lo, hi = pair_lo_hi(mate)
    print("decode G", flush=True)
    pred_g = c24.decode(model, fine, pack["z"], device)
    g_rep = warning_report(
        pred_g, gt.numpy(), pre_fork=pack["pre"], collision=pack["coll"],
        blob_n=pack["blob"], paired=pack["paired"],
    )
    mse = map_mse(pred_g, gt.numpy())
    peaks = pred_g.reshape(len(pred_g), -1).max(1)
    gt_peak = gt.numpy().reshape(len(gt), -1).max(1)

    print("line search D(w Fa+(1-w)Fb) vs y", flush=True)
    w_ls, ls_meta = line_search_w(model, fine, z, gt.numpy(), lo, hi, device, c24)
    print("  line-search w hist", ls_meta["counts"], "vertex", f"{ls_meta['frac_vertex']:.3f}", flush=True)

    plans_w = {
        "pair-mean F": torch.full((len(lo),), 0.5),
        "inverse-MSE (Gauss-Markov)": inverse_mse_weights(mse, lo, hi),
        "copy lower-MSE view": better_view_w(mse, lo, hi),
        "peak-softmax T=0.15 (hack)": peak_softmax_w(peaks, lo, hi, 0.15),
        "line-search min MSE(D(F_w), y)": w_ls,
    }
    weight_rows = []
    results = []
    # G first
    row = c24.score_plan("G", pred_g, pack, None)
    results.append(row)
    h = row["headline"]
    print(
        f"decode G  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} "
        f"rec5={h['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f}",
        flush=True,
    )
    for name, w_lo in plans_w.items():
        weight_rows.append(weight_stats(name, w_lo, mse, peaks, gt_peak, lo, hi))
        f_star = mix_pairs(fine, lo, hi, w_lo)
        print("decode", name, flush=True)
        pred = c24.decode(model, f_star, z, device)
        row = c24.score_plan(name, pred, pack, g_rep)
        results.append(row)
        h = row["headline"]
        feas = row.get("feasible_vs_g") or {}
        mark = " FEASIBLE" if feas.get("feasible") else " no"
        print(
            f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} pre={h['pre_fork_stop_recall']:.3f} "
            f"rec5={h['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
            f"night={row['lighting_stop'].get('night', {}).get('stop_recall')} "
            f"glare={row['lighting_stop'].get('harsh_glare', {}).get('stop_recall')}{mark}",
            flush=True,
        )
    out = {
        "n": len(pred_g),
        "n_pairs": pack["n_pairs"],
        "line_search": ls_meta,
        "weight_stats": weight_rows,
        "results": results,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
