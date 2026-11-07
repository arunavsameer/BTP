#!/usr/bin/env python3
"""Test causal/spurious *splits* of F (not spatial cell masks). Frozen G decoder.

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


def _load24():
    p = Path(__file__).with_name("24_cadet_oracles.py")
    spec = importlib.util.spec_from_file_location("cadet24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pair_delta_channel_score(fine: torch.Tensor, mate: np.ndarray) -> np.ndarray:
    """Mean |GAP(F)-GAP(F_mate)| per channel over complete pairs."""
    acc = torch.zeros(fine.size(1))
    n = 0
    for i, j in enumerate(mate):
        if j < 0 or i >= int(j):
            continue
        d = (fine[i].mean(dim=(1, 2)) - fine[int(j)].mean(dim=(1, 2))).abs()
        acc += d
        n += 1
    return (acc / max(n, 1)).numpy()


def project_out_cell(fine: torch.Tensor, mate: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-cell: F_C = F - proj_{F-Fmate} F. F_S = that projection."""
    idx = torch.from_numpy(np.where(mate >= 0, mate, np.arange(len(mate))))
    delta = fine - fine[idx]
    nrm = delta.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dn = delta / nrm
    coef = (fine * dn).sum(dim=1, keepdim=True)
    f_s = coef * dn
    has = torch.from_numpy(mate >= 0).view(-1, 1, 1, 1).to(fine.dtype)
    f_s = f_s * has
    f_c = fine - f_s
    return f_c, f_s


def project_out_channel(fine: torch.Tensor, mate: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """One 128-D style direction per window (GAP of pair delta), removed everywhere spatially."""
    idx = torch.from_numpy(np.where(mate >= 0, mate, np.arange(len(mate))))
    d = fine.mean(dim=(2, 3)) - fine[idx].mean(dim=(2, 3))
    dn = d / d.norm(dim=1, keepdim=True).clamp_min(1e-6)
    dn = dn[:, :, None, None]
    coef = (fine * dn).sum(dim=1, keepdim=True)
    f_s = coef * dn
    has = torch.from_numpy(mate >= 0).view(-1, 1, 1, 1).to(fine.dtype)
    f_s = f_s * has
    return fine - f_s, f_s


def zero_channels(fine: torch.Tensor, score: np.ndarray, k: int, *, highest: bool) -> torch.Tensor:
    order = np.argsort(score)
    drop = order[-k:] if highest else order[:k]
    out = fine.clone()
    out[:, drop] = 0
    return out


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/split_oracles.json")
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
    pred_g = c24.decode(model, fine, pack["z"], device)
    g_rep = warning_report(
        pred_g, pack["gt"].numpy(), pre_fork=pack["pre"], collision=pack["coll"],
        blob_n=pack["blob"], paired=pack["paired"],
    )
    ch_score = pair_delta_channel_score(fine, mate)
    print("top pair-varying channels", np.argsort(ch_score)[-8:].tolist(), flush=True)
    f_c_cell, f_s_cell = project_out_cell(fine, mate)
    f_c_ch, f_s_ch = project_out_channel(fine, mate)
    peaks = torch.from_numpy(pred_g.reshape(len(pred_g), -1).max(1))
    stronger = peaks[idx] > peaks
    mix = 0.5 * (fine + f_mate)
    f_asymm = fine.clone()
    f_asymm[stronger] = mix[stronger]

    plans = {
        "G": fine,
        "causal = pair-mean F": mix,
        "spurious = F - F_mate": fine - f_mate,
        "causal = remove per-cell pair-delta": f_c_cell,
        "spurious = per-cell pair-delta": f_s_cell,
        "causal = remove GAP pair-delta dir": f_c_ch,
        "spurious = GAP pair-delta dir": f_s_ch,
        "zero 8 most-varying channels": zero_channels(fine, ch_score, 8, highest=True),
        "zero 32 most-varying channels": zero_channels(fine, ch_score, 32, highest=True),
        "keep 64 most-stable channels (zero rest)": zero_channels(fine, ch_score, 64, highest=True),
        "F pulled halfway to stronger mate": f_asymm,
        "AdaIN whole F (repeat)": c24.adain_full(fine, f_mate),
    }
    results = []
    for name, f_star in plans.items():
        print("decode", name, flush=True)
        pred = pred_g if name == "G" else c24.decode(model, f_star, pack["z"], device)
        row = c24.score_plan(name, pred, pack, g_rep if name != "G" else None)
        results.append(row)
        h = row["headline"]
        feas = row.get("feasible_vs_g") or {}
        mark = " FEASIBLE" if feas.get("feasible") else (" no" if feas else "")
        print(
            f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} pre={h['pre_fork_stop_recall']:.3f} "
            f"rec5={h['recall_at_5fa']:.3f} flip={row['flips']['warn_flip']:.3f} "
            f"night={row['lighting_stop'].get('night', {}).get('stop_recall')}{mark}",
            flush=True,
        )
    out = {"n": len(pred_g), "n_pairs": pack["n_pairs"], "channel_pair_l1": ch_score.tolist(), "results": results}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
