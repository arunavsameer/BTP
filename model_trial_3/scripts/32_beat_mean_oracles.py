#!/usr/bin/env python3
"""Oracles aimed at beating pair-mean: occupancy-weighted / local / robust n>2.

Unweighted SVD/mean is wallpaper-dominated. These put the common component on
the GT blob, mix per-cell by local structure, or take a neighborhood median.
No lighting labels, no G ŷ. Does not train or overwrite G.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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


def _load24():
    p = Path(__file__).with_name("24_cadet_oracles.py")
    spec = importlib.util.spec_from_file_location("cadet24", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pair_lo_hi(mate):
    i = torch.arange(len(mate))
    m = torch.as_tensor(mate)
    ok = (m >= 0) & (i < m)
    return i[ok], m[ok]


def scatter_same(fine, lo, hi, c):
    out = fine.clone()
    out[lo] = c
    out[hi] = c
    return out


def batched_rank1(x, chunk=256):
    parts = []
    for i in range(0, x.size(0), chunk):
        sl = x[i : i + chunk].float()
        u, s, vh = torch.linalg.svd(sl, full_matrices=False)
        parts.append((u[:, :, :1] * s[:, :1, None]) @ vh[:, :1, :])
    return torch.cat(parts, 0)


def softmax_mix(fa, fb, score_a, score_b, temp):
    """score [P,1,H,W] or [P,1,1,1]. Same F_c for both views."""
    logits = torch.stack([score_a, score_b], 1) / max(float(temp), 1e-4)
    w = torch.softmax(logits, dim=1)
    return w[:, 0] * fa + w[:, 1] * fb


def winner_mix(fa, fb, score_a, score_b):
    pick_a = score_a >= score_b
    return torch.where(pick_a, fa, fb)


def blob_energy(f, mask):
    return (f.pow(2) * mask).sum(dim=(1, 2, 3), keepdim=True).sqrt()


def print_rec(name, row):
    h = row["headline"]
    feas = row.get("feasible_vs_g") or {}
    mark = ""
    if feas:
        mark = " FEASIBLE" if feas.get("feasible") else " no"
    print(
        f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} "
        f"pre={h['pre_fork_stop_recall']:.3f} rec5={h['recall_at_5fa']:.3f} "
        f"post={h['post_fork_stop_recall']:.3f} small={h['small_blob_stop_recall']:.3f} "
        f"flip={row['flips']['warn_flip']:.3f}{mark}",
        flush=True,
    )


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/beat_mean_oracles.json")
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
    fine = pack["fine"]
    lo, hi = pair_lo_hi(pack["mate"])
    fa, fb = fine[lo], fine[hi]
    mix = 0.5 * (fa + fb)
    p, c, h, w = fa.shape
    gt = pack["gt"][lo]  # pair y ≈ mate y
    y16 = F.interpolate(gt.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).clamp(0, 1)
    ydil = F.interpolate(
        F.max_pool2d(gt.unsqueeze(1), 3, stride=1, padding=1),
        size=(h, w),
        mode="bilinear",
        align_corners=False,
    ).clamp(0, 1)
    blob = (y16 >= 0.5).float()
    empty = blob.flatten(1).sum(1) == 0
    blob[empty] = 0
    bg = 1.0 - blob
    dil = (ydil >= 0.25).float()
    dil[empty] = 0
    bgd = 1.0 - dil

    print("masked rank-1 / local mixes / neighborhood median", flush=True)
    # 1. SVD on blob-masked maps, paste onto mean bg
    ma, mb = fa * blob, fb * blob
    x_blob = torch.stack([ma.reshape(p, -1), mb.reshape(p, -1)], 1)
    l_blob = batched_rank1(x_blob)
    lbm = 0.5 * (l_blob[:, 0] + l_blob[:, 1]).view_as(fa)
    svd_blob_bgmean = lbm * blob + mix * bg
    # stronger L on blob
    na = l_blob[:, 0].norm(dim=1)
    nb = l_blob[:, 1].norm(dim=1)
    lstr = torch.where((na >= nb).view(p, 1, 1, 1), l_blob[:, 0].view_as(fa), l_blob[:, 1].view_as(fa))
    svd_blob_strong = lstr * blob + mix * bg

    # 2. sqrt(y)-weighted full SVD (wallpaper down-weighted)
    sw = y16.sqrt()
    xw = torch.stack([(fa * sw).reshape(p, -1), (fb * sw).reshape(p, -1)], 1)
    lw = batched_rank1(xw)
    lwm = 0.5 * (lw[:, 0] + lw[:, 1]).view_as(fa)
    # unweight approximately: where y>0, L/sqrt(y), else mean
    svd_ywt = torch.where(y16 > 1e-4, lwm / sw.clamp_min(1e-4), mix)

    # 3. dilated-blob SVD
    md = fa * dil
    xdil = torch.stack([md.reshape(p, -1), (fb * dil).reshape(p, -1)], 1)
    ld = batched_rank1(xdil)
    ldm = 0.5 * (ld[:, 0] + ld[:, 1]).view_as(fa)
    svd_dil = ldm * dil + mix * bgd

    # 4. local energy / std mixes (shared F_c)
    ea = fa.norm(dim=1, keepdim=True)
    eb = fb.norm(dim=1, keepdim=True)
    sa = fa.std(dim=1, keepdim=True)
    sb = fb.std(dim=1, keepdim=True)
    t_e = 0.3 * (0.5 * (ea + eb)).median().clamp_min(1e-3)
    t_s = 0.3 * (0.5 * (sa + sb)).median().clamp_min(1e-3)
    loc_e_soft = softmax_mix(fa, fb, ea, eb, float(t_e))
    loc_e_hard = winner_mix(fa, fb, ea, eb)
    loc_s_soft = softmax_mix(fa, fb, sa, sb, float(t_s))
    loc_s_hard = winner_mix(fa, fb, sa, sb)
    # blob-only local winner, bg mean
    e_blob_a = blob_energy(fa, blob)
    e_blob_b = blob_energy(fb, blob)
    win_blob = winner_mix(fa, fb, e_blob_a, e_blob_b)
    blob_win_bgmean = win_blob * blob + mix * bg
    std_blob_a = (fa.std(dim=1, keepdim=True) * blob).sum(dim=(2, 3), keepdim=True)
    std_blob_b = (fb.std(dim=1, keepdim=True) * blob).sum(dim=(2, 3), keepdim=True)
    win_std = winner_mix(fa, fb, std_blob_a, std_blob_b)
    std_win_bgmean = win_std * blob + mix * bg

    # 5. residual shrink only off-blob: shared C = mix, but blob takes max-abs
    maxabs = torch.where(fa.abs() >= fb.abs(), fa, fb)
    blob_maxabs_bgmean = maxabs * blob + mix * bg
    dil_maxabs_bgmean = maxabs * dil + mix * bgd

    # 6. scale-match blob energy then mean (avoid dropout dragging the scale)
    ea_b = blob_energy(fa, blob).clamp_min(1e-6)
    eb_b = blob_energy(fb, blob).clamp_min(1e-6)
    target = 0.5 * (ea_b + eb_b)
    # don't explode empty
    fa_s = torch.where(empty.view(p, 1, 1, 1), fa, fa * target / ea_b)
    fb_s = torch.where(empty.view(p, 1, 1, 1), fb, fb * target / eb_b)
    scale_match_mean = 0.5 * (fa_s + fb_s)

    # 7. neighborhood median across both views' 3×3 (18 samples / channel)
    ua = F.unfold(fa, 3, padding=1).view(p, c, 9, h * w)
    ub = F.unfold(fb, 3, padding=1).view(p, c, 9, h * w)
    neigh = torch.cat([ua, ub], dim=2)
    med = neigh.median(dim=2).values.view(p, c, h, w)
    # neighborhood mean (control)
    nmean = neigh.mean(dim=2).view(p, c, h, w)

    # 8. large-diff on blob → max-abs, else mean
    delta = (fa - fb).abs().mean(dim=1, keepdim=True)
    thr = delta.median()
    disagree = (delta > thr).float() * blob
    hybrid_diff = disagree * maxabs + (1.0 - disagree) * mix

    # 9. softmax mix on blob energy (global per map), bg mean
    t_b = 0.3 * (0.5 * (e_blob_a + e_blob_b)).median().clamp_min(1e-3)
    soft_blob = softmax_mix(fa, fb, e_blob_a, e_blob_b, float(t_b))
    soft_blob_bgmean = soft_blob * blob + mix * bg
    hard_blob_all = winner_mix(fa, fb, e_blob_a, e_blob_b)  # whole map from blob-winner

    # 10. channel-wise softmax on |mean on blob|
    ca = (fa * blob).mean(dim=(2, 3), keepdim=True).abs()
    cb = (fb * blob).mean(dim=(2, 3), keepdim=True).abs()
    ch_soft = softmax_mix(fa, fb, ca, cb, 0.3 * float((0.5 * (ca + cb)).median().clamp_min(1e-4)))

    plans = {
        "G": fine,
        "pair-mean": scatter_same(fine, lo, hi, mix),
        "SVD on blob cells, mean wallpaper": scatter_same(fine, lo, hi, svd_blob_bgmean),
        "SVD blob stronger-L, mean wallpaper": scatter_same(fine, lo, hi, svd_blob_strong),
        "sqrt(y)-weighted SVD": scatter_same(fine, lo, hi, svd_ywt),
        "SVD on dilated blob, mean rest": scatter_same(fine, lo, hi, svd_dil),
        "per-cell softmax ||F||": scatter_same(fine, lo, hi, loc_e_soft),
        "per-cell winner ||F||": scatter_same(fine, lo, hi, loc_e_hard),
        "per-cell softmax channel-std": scatter_same(fine, lo, hi, loc_s_soft),
        "per-cell winner channel-std": scatter_same(fine, lo, hi, loc_s_hard),
        "blob-energy winner, mean wallpaper": scatter_same(fine, lo, hi, blob_win_bgmean),
        "blob channel-std winner, mean wallpaper": scatter_same(fine, lo, hi, std_win_bgmean),
        "max-|F| on blob, mean wallpaper": scatter_same(fine, lo, hi, blob_maxabs_bgmean),
        "max-|F| on dilated blob, mean rest": scatter_same(fine, lo, hi, dil_maxabs_bgmean),
        "scale-match blob energy then mean": scatter_same(fine, lo, hi, scale_match_mean),
        "3x3×2 neighborhood median": scatter_same(fine, lo, hi, med),
        "3x3×2 neighborhood mean": scatter_same(fine, lo, hi, nmean),
        "on blob if pair-diff large: max-|F| else mean": scatter_same(fine, lo, hi, hybrid_diff),
        "softmax blob-energy mix, mean wallpaper": scatter_same(fine, lo, hi, soft_blob_bgmean),
        "whole map from blob-energy winner": scatter_same(fine, lo, hi, hard_blob_all),
        "channel softmax |blob GAP|": scatter_same(fine, lo, hi, ch_soft),
    }

    print("decode G", flush=True)
    pred_g = c24.decode(model, fine, pack["z"], device)
    g_rep = warning_report(
        pred_g, pack["gt"].numpy(), pre_fork=pack["pre"], collision=pack["coll"],
        blob_n=pack["blob"], paired=pack["paired"],
    )
    results = []
    for name, f_star in plans.items():
        print("decode", name, flush=True)
        pred = pred_g if name == "G" else c24.decode(model, f_star, pack["z"], device)
        row = c24.score_plan(name, pred, pack, g_rep if name != "G" else None)
        results.append(row)
        print_rec(name, row)

    slim = []
    mean_stop = None
    for row in results:
        h = row["headline"]
        slim.append({
            "name": row["name"],
            "stop_recall": h["stop_recall"],
            "safe_recall": h["safe_recall"],
            "pre_fork_stop_recall": h["pre_fork_stop_recall"],
            "post_fork_stop_recall": h["post_fork_stop_recall"],
            "recall_at_5fa": h["recall_at_5fa"],
            "small_blob_stop_recall": h["small_blob_stop_recall"],
            "warn_flip": row["flips"]["warn_flip"],
            "feasible": (row.get("feasible_vs_g") or {}).get("feasible"),
        })
        if row["name"] == "pair-mean":
            mean_stop = h["stop_recall"]
            mean_rec5 = h["recall_at_5fa"]
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"results": slim}, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    print("\nBEATS MEAN (STOP>mean and rec5>=mean-0.005)", flush=True)
    for r in sorted(slim, key=lambda x: (x["stop_recall"], x["recall_at_5fa"]), reverse=True):
        beat = mean_stop is not None and r["stop_recall"] > mean_stop + 1e-6 and r["recall_at_5fa"] >= mean_rec5 - 0.005
        flag = " **BEATS MEAN**" if beat else ""
        print(
            f"  {r['stop_recall']:.3f} rec5={r['recall_at_5fa']:.3f} SAFE={r['safe_recall']:.3f} "
            f"pre={r['pre_fork_stop_recall']:.3f}  {r['name']}{flag}",
            flush=True,
        )


if __name__ == "__main__":
    main()
