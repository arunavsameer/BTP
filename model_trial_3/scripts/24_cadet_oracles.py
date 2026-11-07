#!/usr/bin/env python3
"""Frozen-G CaDeT oracles. No training. Does not overwrite G.

Applies CaDeT operations on cached adapter features F, then G's heatmap decoder.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset, build_dataloader  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import feasible_vs_g, warning_report  # noqa: E402
from collision_jepa3.training.engine import move_batch  # noqa: E402
from collision_jepa3.u1_runtime import load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402

CAUTION, STOP = 0.25, 0.60


def _pair_key(cid: str) -> tuple[str, str]:
    parts = str(cid).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}|{parts[1]}", parts[2]
    return str(cid), "?"


def _warn(p: float) -> str:
    if p >= STOP:
        return "STOP"
    if p >= CAUTION:
        return "THREAT"
    return "SAFE"


def masked_stats(x: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [B,C,H,W], w [B,1,H,W] → mu, std [B,C,1,1] over masked cells."""
    wsum = w.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mu = (x * w).sum(dim=(2, 3), keepdim=True) / wsum
    var = ((x - mu) ** 2 * w).sum(dim=(2, 3), keepdim=True) / wsum
    std = var.clamp_min(1e-8).sqrt()
    return mu, std


def adain_masked(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Restyle x toward y's (μ,σ) on the support of w; leave other cells as x."""
    mu_x, std_x = masked_stats(x, w)
    mu_y, std_y = masked_stats(y, w)
    stylized = std_y * (x - mu_x) / std_x.clamp_min(1e-6) + mu_y
    return w * stylized + (1.0 - w) * x


def adain_full(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
    return adain_masked(x, y, ones)


@torch.no_grad()
def collect(model, dataset, cfg, device: str, records_by_id: dict) -> dict:
    cfg.set("training.num_workers", 0)
    loader = build_dataloader(dataset, cfg, shuffle=False)
    model.eval()
    fines, gts, zs = [], [], []
    clip_id, lighting, pre, coll, blob, paired, anchor = [], [], [], [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        z, fine = model.encode_pair(tokens=batch["tokens"])
        fines.append(fine.float().cpu())
        zs.append(z.float().cpu())
        gts.append(batch["h_1s"].float().cpu())
        ids = list(batch["clip_id"])
        clip_id.extend(ids)
        lighting.extend([records_by_id[c].lighting or "unknown" for c in ids])
        pre.append(batch["pre_fork"].bool().cpu())
        coll.extend([str(o) == "collision" for o in batch["outcome_type"]])
        if "blob_n" in batch:
            blob.append(batch["blob_n"].cpu())
        paired.append(batch.get("bg_pair_valid", torch.ones(len(ids))).bool().cpu() if torch.is_tensor(batch.get("bg_pair_valid")) else torch.ones(len(ids), dtype=torch.bool))
        anchor.append(batch["anchor"].cpu())
        if len(clip_id) % 1280 == 0:
            print(f"  collected {len(clip_id)}/{len(dataset)}", flush=True)
    fine = torch.cat(fines, 0)
    z = torch.cat(zs, 0)
    gt = torch.cat(gts, 0)
    pre = torch.cat(pre, 0).numpy()
    anchor = torch.cat(anchor, 0).numpy().astype(np.int32)
    n = len(clip_id)
    blob_n = (gt.numpy() >= 0.5).reshape(n, -1).sum(axis=1).astype(np.int16)
    mate = np.full(n, -1, dtype=np.int32)
    buckets = defaultdict(dict)
    for i, cid in enumerate(clip_id):
        pid, bg = _pair_key(cid)
        buckets[(pid, int(anchor[i]))][bg] = i
    n_pairs = 0
    for bgs in buckets.values():
        if "bg_00" in bgs and "bg_01" in bgs:
            i0, i1 = bgs["bg_00"], bgs["bg_01"]
            mate[i0] = i1
            mate[i1] = i0
            n_pairs += 1
    paired_np = mate >= 0
    print(f"fine {tuple(fine.shape)} pairs {n_pairs}", flush=True)
    return {
        "fine": fine,
        "z": z,
        "gt": gt,
        "clip_id": clip_id,
        "lighting": np.array(lighting),
        "pre": pre,
        "coll": np.asarray(coll, dtype=bool),
        "blob": blob_n,
        "paired": paired_np,
        "mate": mate,
        "n_pairs": n_pairs,
        "anchor": anchor,
    }


@torch.no_grad()
def decode(model, fine: torch.Tensor, z: torch.Tensor, device: str, bs: int = 64) -> np.ndarray:
    model.eval()
    hs = []
    for i in range(0, len(fine), bs):
        f = fine[i : i + bs].to(device)
        zz = z[i : i + bs].to(device)
        h, _ = model.decoder.forward_with_logits(f)
        hs.append(h.float().cpu())
    return torch.cat(hs, 0).numpy()


def flip_rate(pred: np.ndarray, mate: np.ndarray) -> dict:
    n_pairs = 0
    n_flip = n_stop = 0
    for i, j in enumerate(mate):
        if j < 0 or i >= j:
            continue
        n_pairs += 1
        w0, w1 = _warn(float(pred[i].max())), _warn(float(pred[j].max()))
        if w0 != w1:
            n_flip += 1
        if (pred[i].max() >= STOP) != (pred[j].max() >= STOP):
            n_stop += 1
    return {
        "n_pairs": n_pairs,
        "warn_flip": n_flip / max(n_pairs, 1),
        "stop_flip": n_stop / max(n_pairs, 1),
    }


def lighting_stop(pred: np.ndarray, gt: np.ndarray, lighting: np.ndarray) -> dict:
    out = {}
    gt_peak = gt.reshape(len(gt), -1).max(1)
    pr_peak = pred.reshape(len(pred), -1).max(1)
    for L in sorted(set(lighting.tolist())):
        m = lighting == L
        stop = m & (gt_peak >= STOP)
        if stop.sum() == 0:
            continue
        out[L] = {
            "n_stop": int(stop.sum()),
            "stop_recall": float((pr_peak[stop] >= STOP).mean()),
            "mean_peak": float(pr_peak[stop].mean()),
        }
    return out


def score_plan(name: str, pred: np.ndarray, pack: dict, g_report: dict | None) -> dict:
    gt = pack["gt"].numpy()
    rep = warning_report(
        pred,
        gt,
        pre_fork=pack["pre"],
        collision=pack["coll"],
        blob_n=pack["blob"],
        paired=pack["paired"],
    )
    row = {
        "name": name,
        "headline": {
            **rep["headline"],
            "mse": float(rep["all"]["operating"]["mse"]),
        },
        "flips": flip_rate(pred, pack["mate"]),
        "lighting_stop": lighting_stop(pred, gt, pack["lighting"]),
    }
    if g_report is not None:
        row["feasible_vs_g"] = feasible_vs_g(rep, g_report)
    return row


def apply_plans(fine: torch.Tensor, gt: torch.Tensor, pred_g: np.ndarray, mate: np.ndarray) -> dict[str, torch.Tensor]:
    h16 = F.interpolate(gt.unsqueeze(1), size=fine.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    mc = (h16 >= 0.10).float()
    # windows with no occupied cell: keep all-ones so we do not zero the map
    empty = mc.flatten(1).sum(dim=1) == 0
    mc[empty] = 1.0
    ms = 1.0 - mc
    p16 = F.interpolate(
        torch.from_numpy(pred_g).unsqueeze(1), size=fine.shape[-2:], mode="bilinear", align_corners=False
    ).clamp(0, 1)
    mc_p = (p16 >= CAUTION).float()
    empty_p = mc_p.flatten(1).sum(dim=1) == 0
    mc_p[empty_p] = 1.0
    ms_p = 1.0 - mc_p

    mate_idx = torch.from_numpy(np.where(mate >= 0, mate, np.arange(len(mate))))
    f_mate = fine[mate_idx]

    plans = {
        "G": fine,
        "CaDeT-inf oracle Pc (GT cells)": fine * mc,
        "CaDeT Ps only (GT)": fine * ms,
        "CaDeT-inf Pc (G's own STOP/THREAT cells)": fine * mc_p,
        "CaDeT swap Ps cellwise": fine * mc + f_mate * ms,
        "CaDeT do(Ps) AdaIN on spurious cells": adain_masked(fine, f_mate, ms),
        "CaDeT AdaIN whole F (not in paper, control)": adain_full(fine, f_mate),
        "AdaIN on Pc cells (lighting on walker)": adain_masked(fine, f_mate, mc),
        "CaDeT do(Ps) AdaIN + keep Pc": mc * fine + ms * adain_masked(fine, f_mate, ms),
    }
    return plans, mc, ms


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/oracles.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
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
    pack = collect(model, ds, cfg, device, recs)
    print("decode G", flush=True)
    pred_g = decode(model, pack["fine"], pack["z"], device)
    g_rep = warning_report(
        pred_g,
        pack["gt"].numpy(),
        pre_fork=pack["pre"],
        collision=pack["coll"],
        blob_n=pack["blob"],
        paired=pack["paired"],
    )
    plans, _mc, _ms = apply_plans(pack["fine"], pack["gt"], pred_g, pack["mate"])
    results = []
    for name, f_star in plans.items():
        print("decode", name, flush=True)
        pred = pred_g if name == "G" else decode(model, f_star, pack["z"], device)
        row = score_plan(name, pred, pack, g_rep if name != "G" else None)
        results.append(row)
        h = row["headline"]
        feas = row.get("feasible_vs_g")
        mark = ""
        if feas:
            mark = " FEASIBLE" if feas.get("feasible") else " no"
        print(
            f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} pre={h['pre_fork_stop_recall']:.3f} "
            f"rec5={h['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
            f"night={row['lighting_stop'].get('night', {}).get('stop_recall')}{mark}",
            flush=True,
        )
    out = {
        "hardware": describe_hardware(device),
        "n": len(pred_g),
        "n_pairs": pack["n_pairs"],
        "results": results,
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
