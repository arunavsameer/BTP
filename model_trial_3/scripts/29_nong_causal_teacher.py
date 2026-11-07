#!/usr/bin/env python3
"""Causal teachers that do NOT use G's ŷ to pick the view.

Selection uses only: invariant y, F on the GT blob, lighting metadata.
D(F) is used only to score, and for the old G-heuristic baseline.
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import warning_report  # noqa: E402
from collision_jepa3.u1_runtime import load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402

STOP, CAUTION = 0.60, 0.25
# Photometry, not G's STOP-recall. Higher = more light on the scene.
LIGHT_RANK = {
    "noon": 5,
    "dawn": 4,
    "harsh_glare": 3,
    "overcast": 2,
    "dusk": 1,
    "night": 0,
}


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


def upsample_gt(gt: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(gt.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).clamp(0, 1)


def masked_mean(energy: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    """energy [N,H,W], mask [N,1,H,W] → [N] mean energy on mask; 0 if empty."""
    w = mask.squeeze(1)
    num = (energy * w).flatten(1).sum(1)
    den = w.flatten(1).sum(1).clamp_min(1.0)
    out = (num / den).numpy()
    empty = w.flatten(1).sum(1).numpy() <= 0
    out[empty] = 0.0
    return out


def cosine_energy_y(energy: torch.Tensor, y16: torch.Tensor) -> np.ndarray:
    a = energy.flatten(1)
    b = y16.squeeze(1).flatten(1)
    a = a / a.norm(dim=1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return (a * b).sum(1).numpy()


def pick_hi(score: np.ndarray, lo, hi) -> np.ndarray:
    """w_lo = 1 if score[lo] >= score[hi] else 0."""
    return (score[lo] >= score[hi]).astype(np.float32)


def pick_lo_score(score: np.ndarray, lo, hi) -> np.ndarray:
    """w_lo = 1 if score[lo] <= score[hi] (lower is better)."""
    return (score[lo] <= score[hi]).astype(np.float32)


def regime_w(gt_peak, lo, hi, *, stop_score=None, safe_score=None, stop_low=False, safe_low=False):
    """Default w=0.5. On STOP/SAFE pairs, hard-pick using the given scores."""
    w = np.full(len(lo), 0.5, dtype=np.float32)
    stop = (gt_peak[lo] >= STOP) | (gt_peak[hi] >= STOP)
    safe = (gt_peak[lo] < CAUTION) & (gt_peak[hi] < CAUTION)
    if stop_score is not None:
        pick = pick_lo_score if stop_low else pick_hi
        w[stop] = pick(stop_score, lo[stop], hi[stop])
    if safe_score is not None:
        pick = pick_lo_score if safe_low else pick_hi
        w[safe] = pick(safe_score, lo[safe], hi[safe])
    return torch.from_numpy(w)


def blob_pred(pred, gt):
    occ = gt >= 0.5
    out = pred.reshape(len(pred), -1).max(1).astype(np.float64)
    p = pred.reshape(len(pred), -1)
    g = occ.reshape(len(gt), -1)
    n = g.sum(1)
    has = n > 0
    out[has] = (p * g).sum(1)[has] / np.clip(n[has], 1, None)
    return out


def agree(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float | None:
    if not mask.any():
        return None
    return float(np.mean(((a >= 0.5) == (b >= 0.5))[mask]))


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/nong_causal_teacher.json")
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
    gt = pack["gt"]
    gt_np = gt.numpy()
    gt_peak = gt_np.reshape(len(gt), -1).max(1)
    peaks = pred_g.reshape(len(pred_g), -1).max(1)
    y16 = upsample_gt(gt, fine.shape[-2:])
    blob_m = (y16 >= 0.5).float()
    empty = blob_m.flatten(1).sum(1) == 0
    bg_m = 1.0 - blob_m
    energy = fine.norm(dim=1)
    e_blob = masked_mean(energy, blob_m)
    e_bg = masked_mean(energy, bg_m)
    e_all = energy.mean(dim=(1, 2)).numpy()
    e_contrast = e_blob - e_bg
    e_cos = cosine_energy_y(energy, y16)
    e_cos[empty.numpy()] = 0.0
    light = np.array([float(LIGHT_RANK.get(str(L), 2)) for L in pack["lighting"]], dtype=np.float64)
    vis_g = blob_pred(pred_g, gt_np)

    stop = (gt_peak[lo] >= STOP) | (gt_peak[hi] >= STOP)
    safe = (gt_peak[lo] < CAUTION) & (gt_peak[hi] < CAUTION)
    w_g = regime_w(gt_peak, lo, hi, stop_score=vis_g, safe_score=peaks, safe_low=True)
    w_fblob = regime_w(gt_peak, lo, hi, stop_score=e_blob)
    w_fcon = regime_w(gt_peak, lo, hi, stop_score=e_contrast)
    w_fcos = regime_w(gt_peak, lo, hi, stop_score=e_cos)
    w_lit = regime_w(gt_peak, lo, hi, stop_score=light)
    w_f_both = regime_w(gt_peak, lo, hi, stop_score=e_blob, safe_score=e_all, safe_low=True)
    w_lit_fquiet = regime_w(gt_peak, lo, hi, stop_score=light, safe_score=e_all, safe_low=True)
    w_fcon_both = regime_w(gt_peak, lo, hi, stop_score=e_contrast, safe_score=e_all, safe_low=True)

    # Spatial: on STOP, occupancy cells from higher F-blob-energy view, wallpaper = mean.
    w_stop_f = pick_hi(e_blob, lo, hi)
    f_pick = mix_pairs(fine, lo, hi, torch.from_numpy(w_stop_f))
    mix = 0.5 * (fine + fine[torch.from_numpy(np.where(mate >= 0, mate, np.arange(len(mate))))])
    f_gate = mix.clone()
    bm = blob_m
    # apply gated mix only on STOP pair members
    stop_idx = np.unique(np.concatenate([lo[stop], hi[stop]]))
    if len(stop_idx):
        si = torch.from_numpy(stop_idx)
        f_gate[si] = bm[si] * f_pick[si] + (1.0 - bm[si]) * mix[si]

    plans_w = {
        "pair-mean": torch.full((len(lo),), 0.5),
        "G-heuristic two-regime (uses ŷ)": w_g,
        "STOP: F-energy on GT blob, else mean": w_fblob,
        "STOP: F-contrast blob-vs-bg, else mean": w_fcon,
        "STOP: F-energy map cosine y, else mean": w_fcos,
        "STOP: brighter lighting, else mean": w_lit,
        "STOP: F-blob energy, SAFE: lower ||F||": w_f_both,
        "STOP: F-contrast, SAFE: lower ||F||": w_fcon_both,
        "STOP: brighter lighting, SAFE: lower ||F||": w_lit_fquiet,
    }
    agree_row = {
        "n_stop_pairs": int(stop.sum()),
        "n_safe_pairs": int(safe.sum()),
        "stop_Fblob_vs_Gvis": agree(pick_hi(e_blob, lo, hi), pick_hi(vis_g, lo, hi), stop),
        "stop_Fcontrast_vs_Gvis": agree(pick_hi(e_contrast, lo, hi), pick_hi(vis_g, lo, hi), stop),
        "stop_Fcos_vs_Gvis": agree(pick_hi(e_cos, lo, hi), pick_hi(vis_g, lo, hi), stop),
        "stop_light_vs_Gvis": agree(pick_hi(light, lo, hi), pick_hi(vis_g, lo, hi), stop),
        "stop_Fblob_vs_light": agree(pick_hi(e_blob, lo, hi), pick_hi(light, lo, hi), stop),
        "safe_Flower_vs_Gquiet": agree(pick_lo_score(e_all, lo, hi), pick_lo_score(peaks, lo, hi), safe),
    }
    print("agreement", json.dumps(agree_row, indent=2), flush=True)

    results = []
    row = c24.score_plan("G", pred_g, pack, None)
    results.append(row)
    h = row["headline"]
    print(
        f"decode G  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} rec5={h['recall_at_5fa']:.3f}",
        flush=True,
    )
    for name, w_lo in plans_w.items():
        print("decode", name, flush=True)
        pred = c24.decode(model, mix_pairs(fine, lo, hi, w_lo), pack["z"], device)
        row = c24.score_plan(name, pred, pack, g_rep)
        results.append(row)
        hh = row["headline"]
        feas = row.get("feasible_vs_g") or {}
        mark = " FEASIBLE" if feas.get("feasible") else " no"
        print(
            f"  STOP={hh['stop_recall']:.3f} SAFE={hh['safe_recall']:.3f} pre={hh['pre_fork_stop_recall']:.3f} "
            f"rec5={hh['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
            f"night={row['lighting_stop'].get('night', {}).get('stop_recall')} "
            f"glare={row['lighting_stop'].get('harsh_glare', {}).get('stop_recall')}{mark}",
            flush=True,
        )
    print("decode STOP blob-cells from F-energy, bg=mean, SAFE=mean", flush=True)
    pred = c24.decode(model, f_gate, pack["z"], device)
    row = c24.score_plan("STOP: blob F-energy, bg mean, SAFE mean", pred, pack, g_rep)
    results.append(row)
    hh = row["headline"]
    feas = row.get("feasible_vs_g") or {}
    mark = " FEASIBLE" if feas.get("feasible") else " no"
    print(
        f"  STOP={hh['stop_recall']:.3f} SAFE={hh['safe_recall']:.3f} pre={hh['pre_fork_stop_recall']:.3f} "
        f"rec5={hh['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
        f"night={row['lighting_stop'].get('night', {}).get('stop_recall')} "
        f"glare={row['lighting_stop'].get('harsh_glare', {}).get('stop_recall')}{mark}",
        flush=True,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"n": len(pred_g), "n_pairs": pack["n_pairs"], "agreement": agree_row, "results": results},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
