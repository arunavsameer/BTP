#!/usr/bin/env python3
"""Frozen oracles for common/residual, RPCA, invariant subspace, CCA, cross-predict.

No lighting labels, no G ŷ selection. Does not train or overwrite G.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

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


def pair_lo_hi(mate):
    i = torch.arange(len(mate))
    mate_t = torch.as_tensor(mate)
    ok = (mate_t >= 0) & (i < mate_t)
    return i[ok], mate_t[ok]


def scatter_pair(fine, lo, hi, a, b):
    """a,b [P,C,H,W] → full N tensor; unpaired stay as G's F."""
    out = fine.clone()
    out[lo] = a
    out[hi] = b
    return out


def scatter_same(fine, lo, hi, c):
    return scatter_pair(fine, lo, hi, c, c)


def batched_rank1(x, chunk=256):
    """x [P,2,D] → L [P,2,D] rank-1 reconstruction."""
    parts = []
    for i in range(0, x.size(0), chunk):
        sl = x[i : i + chunk].float()
        u, s, vh = torch.linalg.svd(sl, full_matrices=False)
        # L = s0 * u[:, :, 0:1] * vh[:, 0:1, :]
        l = (u[:, :, :1] * s[:, :1, None]) @ vh[:, :1, :]
        parts.append(l)
    return torch.cat(parts, 0)


def rpca_rank1(x, lam_scale: float, iters: int = 8, chunk: int = 256):
    """X=L+S, L rank-1, S L1. lam = lam_scale * median(|X|) per pair."""
    p, two, d = x.shape
    s = torch.zeros_like(x)
    l = x.clone()
    med = x.abs().median(dim=2).values.median(dim=1).values.clamp_min(1e-6)  # [P]
    lam = (lam_scale * med).view(-1, 1, 1)
    for _ in range(iters):
        l = batched_rank1(x - s, chunk=chunk)
        z = x - l
        s = torch.sign(z) * torch.relu(z.abs() - lam)
    return l, s


def locations(fine):
    """[N,C,H,W] → [N*H*W, C]"""
    return fine.permute(0, 2, 3, 1).reshape(-1, fine.size(1))


def inv_sqrt(c, eps=1e-4):
    e, v = torch.linalg.eigh(c)
    e = e.clamp_min(eps)
    return (v * e.rsqrt()) @ v.T


def channel_delta_eigs(fa, fb):
    d = locations(fa - fb)
    n = max(d.size(0), 1)
    cov = (d.T @ d) / n
    e, u = torch.linalg.eigh(cov)  # ascending
    return e, u


def project_channels(fine, u_keep):
    """F' = U_keep U_keep^T F at each location. u_keep [C,k]."""
    c, h, w = fine.size(1), fine.size(2), fine.size(3)
    x = fine.permute(0, 2, 3, 1).reshape(-1, c)
    y = (x @ u_keep) @ u_keep.T
    return y.view(fine.size(0), h, w, c).permute(0, 3, 1, 2).contiguous()


def cca_project(fa, fb, m: int):
    """Project both views onto top-m CCA subspace (view-a weights)."""
    xa, xb = locations(fa), locations(fb)
    n = xa.size(0)
    xa = xa - xa.mean(0)
    xb = xb - xb.mean(0)
    caa = (xa.T @ xa) / n
    cbb = (xb.T @ xb) / n
    cab = (xa.T @ xb) / n
    wa = inv_sqrt(caa) @ torch.linalg.svd(inv_sqrt(caa) @ cab @ inv_sqrt(cbb), full_matrices=False)[0][:, :m]
    # orthonormalize in data space via the m columns
    return project_channels(fa, wa), project_channels(fb, wa), wa


def ridge_mate(fa, fb, lam=1e-2):
    """128×128 W minimizing ||W Fa - Fb|| on locations, and the reverse average."""
    xa, xb = locations(fa), locations(fb)
    n = xa.size(0)
    xtx = xa.T @ xa + xb.T @ xb
    xty = xa.T @ xb + xb.T @ xa
    w = torch.linalg.solve(xtx + lam * n * torch.eye(xa.size(1), device=xa.device), xty)
    return w


def apply_W(fine, w):
    c, h, ww = fine.size(1), fine.size(2), fine.size(3)
    x = fine.permute(0, 2, 3, 1).reshape(-1, c)
    y = x @ w
    return y.view(fine.size(0), h, ww, c).permute(0, 3, 1, 2).contiguous()


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
    parser.add_argument("--out", default="reports/cadet_diag/decomp_oracles.json")
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
    shape = fa.shape
    p, c, h, w = shape
    d = c * h * w
    x = torch.stack([fa.reshape(p, d), fb.reshape(p, d)], dim=1)

    print("rank-1 SVD of [Fa; Fb]", flush=True)
    l1 = batched_rank1(x)
    la, lb = l1[:, 0].view(shape), l1[:, 1].view(shape)
    lmean = 0.5 * (la + lb)
    na = la.reshape(p, -1).norm(dim=1)
    nb = lb.reshape(p, -1).norm(dim=1)
    take_a = (na >= nb).view(p, 1, 1, 1)
    lstrong = torch.where(take_a, la, lb)

    print("RPCA rank-1 + sparse S", flush=True)
    l_lo, _ = rpca_rank1(x, 1.0)
    l_hi, _ = rpca_rank1(x, 3.0)
    rpca_mean_lo = 0.5 * (l_lo[:, 0] + l_lo[:, 1]).view(shape)
    rpca_mean_hi = 0.5 * (l_hi[:, 0] + l_hi[:, 1]).view(shape)

    print("Δ channel eigenspectrum", flush=True)
    evals, u = channel_delta_eigs(fa, fb)
    print("  Δ eig max/median/min", float(evals[-1]), float(evals[evals.numel() // 2]), float(evals[0]), flush=True)
    drop = {k: project_channels(fine, u[:, :-k] if k else u) for k in (1, 8, 16, 32)}
    keep96 = project_channels(fine, u[:, :96])

    print("CCA top-m", flush=True)
    cca32_a, cca32_b, _ = cca_project(fa, fb, 32)
    cca16_a, cca16_b, _ = cca_project(fa, fb, 16)
    cca32 = scatter_pair(fine, lo, hi, 0.5 * (cca32_a + cca32_b), 0.5 * (cca32_a + cca32_b))
    cca16 = scatter_pair(fine, lo, hi, 0.5 * (cca16_a + cca16_b), 0.5 * (cca16_a + cca16_b))
    cca32_own = scatter_pair(fine, lo, hi, cca32_a, cca32_b)

    print("ridge cross-predict F_mate ≈ W F", flush=True)
    wmat = ridge_mate(fa, fb)
    wf = apply_W(fine, wmat)
    idx = torch.as_tensor(pack["mate"])
    idx = torch.where(idx >= 0, idx, torch.arange(len(fine)))
    half_pred = 0.5 * (fine + wf)
    shared_pred = scatter_same(fine, lo, hi, 0.5 * (wf[lo] + wf[hi]))

    # scale-invariant mean
    eps = 1e-6
    naf = fa.reshape(p, -1).norm(dim=1).clamp_min(eps).view(p, 1, 1, 1)
    nbf = fb.reshape(p, -1).norm(dim=1).clamp_min(eps).view(p, 1, 1, 1)
    unit_mean = 0.5 * (fa / naf + fb / nbf)
    scale_mean = unit_mean * (0.5 * (naf + nbf))

    # geometric-mean magnitudes
    sgn = torch.sign(mix)
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    geom = sgn * (fa.abs() * fb.abs()).sqrt()

    # shrink toward mean (soft common): F_c shared
    shrink25 = 0.75 * mix + 0.25 * fa  # not shared
    # shared shrink is just mix. per-clip: (1-α)F + α mean_pair
    def shrink(alpha):
        return scatter_pair(fine, lo, hi, (1 - alpha) * fa + alpha * mix, (1 - alpha) * fb + alpha * mix)

    # joint channel low-rank: U from concat of both views' locations
    print("joint channel PCA r=16", flush=True)
    both = torch.cat([locations(fa), locations(fb)], 0)
    both = both - both.mean(0)
    cov = (both.T @ both) / max(both.size(0), 1)
    _, uj = torch.linalg.eigh(cov)
    joint16 = project_channels(fine, uj[:, -16:])

    plans = {
        "G": fine,
        "1 additive common C=mean": scatter_same(fine, lo, hi, mix),
        "2 soft common shrink α=0.5 (each keeps half self)": shrink(0.5),
        "3 soft common shrink α=0.75": shrink(0.75),
        "4 SVD rank-1 own L row": scatter_pair(fine, lo, hi, la, lb),
        "5 SVD rank-1 shared mean L": scatter_same(fine, lo, hi, lmean),
        "6 SVD rank-1 both get stronger L": scatter_same(fine, lo, hi, lstrong),
        "7 RPCA λ=1 shared L": scatter_same(fine, lo, hi, rpca_mean_lo),
        "8 RPCA λ=3 shared L": scatter_same(fine, lo, hi, rpca_mean_hi),
        "9 intersection min-|F|": scatter_same(fine, lo, hi, torch.where(fa.abs() <= fb.abs(), fa, fb)),
        "10 geometric-mean |F|, sign of mean": scatter_same(fine, lo, hi, geom),
        "11 invariant subspace drop top-1 Δ eig": drop[1],
        "12 drop top-8 Δ eig": drop[8],
        "13 drop top-16 Δ eig": drop[16],
        "14 drop top-32 Δ eig": drop[32],
        "15 keep 96 most-stable Δ eigs": keep96,
        "16 CCA-32 shared reconstruct": cca32,
        "17 CCA-16 shared reconstruct": cca16,
        "18 CCA-32 own-view project": cca32_own,
        "19 ridge F_c=0.5(F+W F) cross-predict": half_pred,
        "20 ridge F_c=W F (predict mate)": wf,
        "21 ridge shared 0.5(WFa+WFb)": shared_pred,
        "22 scale-invariant mean": scatter_same(fine, lo, hi, scale_mean),
        "23 joint channel PCA r=16": joint16,
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
    meta = {
        "delta_eig_top10": [float(x) for x in evals[-10:].flip(0)],
        "ridge_W_frob": float(wmat.norm()),
        "ridge_W_diag_mean": float(wmat.diag().mean()),
        "note": "Val-fit subspace/CCA/ridge are oracles (stats from the same pairs they are applied to).",
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "results": slim}, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    print("\nRANK by STOP then rec5", flush=True)
    for r in sorted(slim, key=lambda x: (x["stop_recall"], x["recall_at_5fa"]), reverse=True):
        print(
            f"  {r['stop_recall']:.3f} rec5={r['recall_at_5fa']:.3f} SAFE={r['safe_recall']:.3f} "
            f"pre={r['pre_fork_stop_recall']:.3f}  {r['name']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
