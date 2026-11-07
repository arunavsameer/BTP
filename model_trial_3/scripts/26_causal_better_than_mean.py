#!/usr/bin/env python3
"""Frozen oracles for causal F that are not pair-mean. Does not train or overwrite G.

Pair-mean F is feasible but pulls strong clips (glare) down. This file tests
targets that keep occupancy from the stronger view and only mix the residual
where it is actually style.
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

# G STOP-recall order: dawn/noon easy, dusk/night hard.
LIGHT_RANK = {
    "dawn": 5,
    "noon": 4,
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


def upsample_map(x: torch.Tensor, hw: tuple[int, int], thresh: float) -> torch.Tensor:
    """x [N,H,W] → binary mask [N,1,h,w]. Empty windows stay all-ones."""
    m = F.interpolate(x.unsqueeze(1), size=hw, mode="bilinear", align_corners=False)
    w = (m >= thresh).float()
    empty = w.flatten(1).sum(dim=1) == 0
    w[empty] = 1.0
    return w


def peak_weighted(fine: torch.Tensor, f_mate: torch.Tensor, peaks: torch.Tensor, idx: torch.Tensor, temp: float) -> torch.Tensor:
    """Softmax over pair peaks. T→∞ is mean, T→0 copies the stronger F onto both."""
    logits = torch.stack([peaks, peaks[idx]], dim=1) / max(temp, 1e-4)
    w = torch.softmax(logits, dim=1)
    return w[:, 0, None, None, None] * fine + w[:, 1, None, None, None] * f_mate


def pull_weak(fine: torch.Tensor, f_mate: torch.Tensor, mate_stronger: torch.Tensor, alpha: float) -> torch.Tensor:
    """Only the weaker clip moves: F_weak ← (1-α) F_weak + α F_strong.

    mate_stronger[i] is True iff the mate's G peak is higher than i's.
    """
    out = fine.clone()
    out[mate_stronger] = (1.0 - alpha) * fine[mate_stronger] + alpha * f_mate[mate_stronger]
    return out


def gated_mix(
    fine: torch.Tensor,
    f_mate: torch.Tensor,
    mix: torch.Tensor,
    gate: torch.Tensor,
    *,
    occ: str,
) -> torch.Tensor:
    """On occupancy cells use `occ` policy; on background use pair-mean.

    occ: 'self' keep each clip's F, 'strong' copy stronger-peak F, 'mean' (control).
    """
    bg = 1.0 - gate
    if occ == "self":
        occ_f = fine
    elif occ == "strong":
        occ_f = torch.where(gate.bool(), mix, fine)  # placeholder, set below
        # stronger F at occupancy: mix is pair-mean; use f_mate where mate is stronger
        # caller should pass f_strong as mix when occ=='strong'
        occ_f = mix
    else:
        occ_f = mix
    return gate * occ_f + bg * mix


def f_strong_map(fine: torch.Tensor, f_mate: torch.Tensor, mate_stronger: torch.Tensor) -> torch.Tensor:
    """Both clips receive the F of the higher G-peak mate."""
    out = fine.clone()
    out[mate_stronger] = f_mate[mate_stronger]
    return out


def loc_winner(fine: torch.Tensor, f_mate: torch.Tensor) -> torch.Tensor:
    """Per 16×16 cell, take the vector with larger channel L2."""
    na = fine.norm(dim=1, keepdim=True)
    nb = f_mate.norm(dim=1, keepdim=True)
    pick_self = na >= nb
    return torch.where(pick_self, fine, f_mate)


def style_to(fine: torch.Tensor, style_src: torch.Tensor, c24) -> torch.Tensor:
    return c24.adain_full(fine, style_src)


def instancenorm(fine: torch.Tensor) -> torch.Tensor:
    mu = fine.mean(dim=(2, 3), keepdim=True)
    std = fine.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return (fine - mu) / std


def lighting_rank_tensor(lighting: np.ndarray) -> torch.Tensor:
    return torch.tensor([float(LIGHT_RANK.get(str(L), 0)) for L in lighting], dtype=torch.float32)


def easy_light_copy(fine: torch.Tensor, idx: torch.Tensor, lighting: np.ndarray) -> torch.Tensor:
    rank = lighting_rank_tensor(lighting)
    take_mate = (rank[idx] > rank).view(-1, 1, 1, 1)
    return torch.where(take_mate, fine[idx], fine)


def lighting_softmax(fine: torch.Tensor, f_mate: torch.Tensor, idx: torch.Tensor, lighting: np.ndarray, temp: float) -> torch.Tensor:
    rank = lighting_rank_tensor(lighting)
    logits = torch.stack([rank, rank[idx]], dim=1) / max(temp, 1e-4)
    w = torch.softmax(logits, dim=1)
    return w[:, 0, None, None, None] * fine + w[:, 1, None, None, None] * f_mate


def build_plans(fine, f_mate, mix, peaks, idx, stronger, pred_g, gt, lighting, c24):
    f_strong = f_strong_map(fine, f_mate, stronger)
    h, w = fine.shape[-2:]
    gt_occ = upsample_map(gt, (h, w), 0.5)
    g_occ = upsample_map(torch.from_numpy(pred_g), (h, w), 0.25)
    mu_mean = 0.5 * (fine.mean(dim=(2, 3), keepdim=True) + f_mate.mean(dim=(2, 3), keepdim=True))
    # AdaIN toward pair-mean stats: share style, keep own spatial content.
    style_mean = style_to(fine, mix, c24)
    # Style-free content mean, restyle to pair-mean μσ.
    content_mean = 0.5 * (instancenorm(fine) + instancenorm(f_mate))
    std_mean = 0.5 * (
        fine.std(dim=(2, 3), keepdim=True) + f_mate.std(dim=(2, 3), keepdim=True)
    ).clamp_min(1e-6)
    restyled_content_mean = std_mean * content_mean + mu_mean
    delta = fine - f_mate
    return {
        "G": fine,
        "pair-mean F": mix,
        "peak-softmax T=1": peak_weighted(fine, f_mate, peaks, idx, 1.0),
        "peak-softmax T=0.3": peak_weighted(fine, f_mate, peaks, idx, 0.3),
        "peak-softmax T=0.15": peak_weighted(fine, f_mate, peaks, idx, 0.15),
        "peak-softmax T=0.1": peak_weighted(fine, f_mate, peaks, idx, 0.1),
        "copy stronger F to both": f_strong,
        "pull weak α=0.5": pull_weak(fine, f_mate, stronger, 0.5),
        "pull weak α=0.75": pull_weak(fine, f_mate, stronger, 0.75),
        "pull weak α=1 (copy strong onto weak)": pull_weak(fine, f_mate, stronger, 1.0),
        "shrink residual α=0.5 (halfway to mean)": fine - 0.25 * delta,
        "occ=self, bg=mean (GT≥0.5)": gt_occ * fine + (1.0 - gt_occ) * mix,
        "occ=strong, bg=mean (GT≥0.5)": gt_occ * f_strong + (1.0 - gt_occ) * mix,
        "occ=self, bg=mean (G THREAT)": g_occ * fine + (1.0 - g_occ) * mix,
        "occ=strong, bg=mean (G THREAT)": g_occ * f_strong + (1.0 - g_occ) * mix,
        "AdaIN each to pair-mean stats": style_mean,
        "IN-mean content + pair-mean style": restyled_content_mean,
        "per-cell L2 winner": loc_winner(fine, f_mate),
        "copy easier-lighting F to both": easy_light_copy(fine, idx, lighting),
        "lighting-softmax T=1": lighting_softmax(fine, f_mate, idx, lighting, 1.0),
        "lighting-softmax T=0.5": lighting_softmax(fine, f_mate, idx, lighting, 0.5),
        "AdaIN weak toward strong": style_to(fine, f_strong, c24),
    }


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/better_mean_oracles.json")
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
    idx = torch.from_numpy(np.where(mate >= 0, mate, np.arange(len(mate))))
    f_mate = fine[idx]
    mix = 0.5 * (fine + f_mate)
    pred_g = c24.decode(model, fine, pack["z"], device)
    g_rep = warning_report(
        pred_g, pack["gt"].numpy(), pre_fork=pack["pre"], collision=pack["coll"],
        blob_n=pack["blob"], paired=pack["paired"],
    )
    peaks = torch.from_numpy(pred_g.reshape(len(pred_g), -1).max(1))
    stronger = peaks[idx] > peaks
    plans = build_plans(
        fine, f_mate, mix, peaks, idx, stronger, pred_g, pack["gt"], pack["lighting"], c24,
    )
    results = []
    for name, f_star in plans.items():
        print("decode", name, flush=True)
        pred = pred_g if name == "G" else c24.decode(model, f_star, pack["z"], device)
        row = c24.score_plan(name, pred, pack, g_rep if name != "G" else None)
        results.append(row)
        h = row["headline"]
        feas = row.get("feasible_vs_g") or {}
        mark = " FEASIBLE" if feas.get("feasible") else (" no" if feas else "")
        glare = row["lighting_stop"].get("harsh_glare", {}).get("stop_recall")
        print(
            f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} pre={h['pre_fork_stop_recall']:.3f} "
            f"rec5={h['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
            f"night={row['lighting_stop'].get('night', {}).get('stop_recall')} "
            f"glare={glare}{mark}",
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
