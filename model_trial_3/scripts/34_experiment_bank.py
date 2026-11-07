#!/usr/bin/env python3
"""Train-fit only, single-clip inference, frozen G decoder. Experiment bank 1–10.

Does not overwrite G. Fits on train pairs; evaluates each val clip independently.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts._common import add_config_args, load_cfg  # noqa: E402

from collision_jepa3.data.dataset import TeacherDataset  # noqa: E402
from collision_jepa3.data.manifest import load_manifest  # noqa: E402
from collision_jepa3.data.splits import load_splits  # noqa: E402
from collision_jepa3.evaluation.counterfactual import warning_report  # noqa: E402
from collision_jepa3.losses.heatmap import soft_bce_logits  # noqa: E402
from collision_jepa3.u1_runtime import load_frozen_g  # noqa: E402
from collision_jepa3.utils.device import describe_hardware, resolve_device  # noqa: E402


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


def upsample_y(gt: torch.Tensor, hw=(16, 16)) -> torch.Tensor:
    return F.interpolate(gt.unsqueeze(1), size=hw, mode="bilinear", align_corners=False).clamp(0, 1)


def loc(fine: torch.Tensor) -> torch.Tensor:
    return fine.permute(0, 2, 3, 1).reshape(-1, fine.size(1))


def cov_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x [N,C] → mean, cov (unbiased=False)."""
    n = max(x.size(0), 1)
    mu = x.mean(0)
    xc = x - mu
    return mu, (xc.T @ xc) / n


def outer(v: torch.Tensor) -> torch.Tensor:
    return torch.outer(v, v)


def gevd(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-3):
    """A v = λ B v. Returns eigenvalues ascending, vectors in original space."""
    c = A.size(0)
    I = torch.eye(c, dtype=A.dtype, device=A.device)
    tr = float(B.trace().clamp_min(1e-6))
    Breg = B + (eps * tr / c) * I
    e, U = torch.linalg.eigh(Breg)
    e = e.clamp_min(eps * tr / c)
    Binvh = (U * e.rsqrt()) @ U.T
    M = 0.5 * (Binvh @ A @ Binvh + (Binvh @ A @ Binvh).T)
    lam, V = torch.linalg.eigh(M)
    vecs = Binvh @ V
    vecs = vecs / vecs.norm(dim=0, keepdim=True).clamp_min(1e-8)
    return lam, vecs


def project_keep(fine: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """F' = U U^T F per location. U [C,k]."""
    c, h, w = fine.size(1), fine.size(2), fine.size(3)
    x = loc(fine)
    y = (x @ U) @ U.T
    return y.view(fine.size(0), h, w, c).permute(0, 3, 1, 2).contiguous()


def project_drop(fine: torch.Tensor, U_drop: torch.Tensor) -> torch.Tensor:
    return fine - project_keep(fine, U_drop)


def task_pair_stats(fine, gt, mate):
    """Train-only S_pair, S_task, S_B, S_W. Channel 128.

    S_B uses occupancy slices so the Fisher/PLS pencils have rank > 1.
    """
    lo, hi = pair_lo_hi(mate)
    fa, fb = fine[lo], fine[hi]
    d = loc(fa - fb)
    _, S_pair = cov_xy(d)
    y16 = upsample_y(gt)
    x = loc(fine)
    y = y16.reshape(-1)
    occ = y >= 0.5
    emp = y < 0.25
    _, S_occ = cov_xy(x[occ] if int(occ.sum()) > 10 else x)
    _, S_all = cov_xy(x)
    edges = (0.0, 0.15, 0.40, 0.70, 1.01)
    mu = x.mean(0)
    S_B = torch.zeros(x.size(1), x.size(1), dtype=x.dtype)
    S_W = torch.zeros_like(S_B)
    n_used = 0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (y >= a) & (y < b)
        n_k = int(m.sum())
        if n_k < 20:
            continue
        xk = x[m]
        muk = xk.mean(0)
        S_B = S_B + n_k * outer(muk - mu)
        _, ck = cov_xy(xk)
        S_W = S_W + ck * n_k
        n_used += n_k
    S_B = S_B / max(n_used, 1)
    S_W = S_W / max(n_used, 1)
    S_task = S_B + 0.25 * S_occ + 0.05 * S_all
    # Rank-1 occupancy contrast, kept for diagnostics
    mu_o = x[occ].mean(0) if int(occ.sum()) > 10 else x.mean(0)
    mu_e = x[emp].mean(0) if int(emp.sum()) > 10 else x.mean(0)
    return {
        "S_pair": S_pair,
        "S_task": S_task,
        "S_B": S_B,
        "S_W": S_W,
        "S_all": S_all,
        "S_B_binary": outer(mu_o - mu_e),
        "n_pairs": len(lo),
        "n_occ": int(occ.sum()),
        "n_emp": int(emp.sum()),
    }


@torch.no_grad()
def apply_print_decode(name, fstar, model, pack, device, c24, g_rep, results):
    print("decode", name, flush=True)
    pred = c24.decode(model, fstar, pack["z"], device)
    row = c24.score_plan(name, pred, pack, g_rep)
    h, ls, fl = row["headline"], row["lighting_stop"], row["flips"]
    feas = row.get("feasible_vs_g") or {}
    mark = " FEASIBLE" if feas.get("feasible") else " no"
    night = ls.get("night", {}).get("stop_recall")
    dusk = ls.get("dusk", {}).get("stop_recall")
    glare = ls.get("harsh_glare", {}).get("stop_recall")
    print(
        f"  STOP={h['stop_recall']:.3f} SAFE={h['safe_recall']:.3f} rec5={h['recall_at_5fa']:.3f} "
        f"flip={fl['warn_flip']:.3f} post={h['post_fork_stop_recall']:.3f} "
        f"night={night} dusk={dusk} glare={glare}{mark}",
        flush=True,
    )
    results.append({
        "name": name,
        "stop_recall": h["stop_recall"],
        "safe_recall": h["safe_recall"],
        "recall_at_5fa": h["recall_at_5fa"],
        "warn_flip": fl["warn_flip"],
        "post_fork_stop_recall": h["post_fork_stop_recall"],
        "pre_fork_stop_recall": h["pre_fork_stop_recall"],
        "night_stop": night,
        "dusk_stop": dusk,
        "glare_stop": glare,
        "feasible": feas.get("feasible"),
    })


def fit_gradients(decoder, fine, gt, device, max_n=4096):
    """Empirical Fisher E[g g^T] and Cov(g), g=dL/dF. Subsample windows."""
    decoder.eval()
    n = min(int(max_n), len(fine))
    idx = torch.randperm(len(fine))[:n]
    acc_gg = torch.zeros(128, 128, device=device)
    acc_g = torch.zeros(128, device=device)
    count = 0
    bs = 8
    with torch.enable_grad():
        for s in range(0, n, bs):
            sl = idx[s : s + bs]
            f = fine[sl].to(device).detach().requires_grad_(True)
            y = gt[sl].to(device)
            _, logits = decoder.forward_with_logits(f)
            loss = soft_bce_logits(logits, y)
            g = torch.autograd.grad(loss, f, retain_graph=False)[0].detach()
            gl = loc(g)
            acc_gg += gl.T @ gl
            acc_g += gl.sum(0)
            count += gl.size(0)
            del f, g, logits, loss
    mu = acc_g / max(count, 1)
    fisher = acc_gg / max(count, 1)
    cov_g = fisher - torch.outer(mu, mu)
    return fisher.cpu(), cov_g.cpu()


class OrthoQ(nn.Module):
    def __init__(self, c=128):
        super().__init__()
        self.A = nn.Parameter(torch.eye(c))

    def Q(self):
        q, r = torch.linalg.qr(self.A)
        s = torch.sign(torch.diag(r))
        s = torch.where(s == 0, torch.ones_like(s), s)
        return q * s.unsqueeze(0)

    def forward(self, f):
        q = self.Q()
        return torch.einsum("oc,bchw->bohw", q, f)


class Coupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1 = nn.Conv2d(64, 64, 1)
        self.n2 = nn.Conv2d(64, 64, 1)
        nn.init.zeros_(self.n1.weight)
        nn.init.zeros_(self.n1.bias)
        nn.init.zeros_(self.n2.weight)
        nn.init.zeros_(self.n2.bias)

    def forward(self, f):
        a, b = f[:, :64], f[:, 64:]
        b = b + self.n1(a)
        a = a + self.n2(b)
        return torch.cat([a, b], 1)


def train_linear_map(module, decoder, train, device, *, epochs=4, lam_inv=0.1, task_ch=96, lr=1e-3):
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=0.0)
    feat, y, mate = train["fine"], train["gt"], train["mate"]
    lo, hi = pair_lo_hi(mate)
    n_p = len(lo)
    for ep in range(epochs):
        module.train()
        decoder.eval()
        perm = np.random.permutation(n_p)
        tot = 0.0
        nb = 0
        for s in range(0, n_p, 32):
            sl = perm[s : s + 32]
            fa = feat[lo[sl]].to(device)
            fb = feat[hi[sl]].to(device)
            ya = y[lo[sl]].to(device)
            yb = y[hi[sl]].to(device)
            f = torch.cat([fa, fb], 0)
            tgt = torch.cat([ya, yb], 0)
            fp = module(f)
            _, logits = decoder.forward_with_logits(fp)
            lh = soft_bce_logits(logits, tgt)
            pa, pb = fp.chunk(2)
            if task_ch is not None:
                linv = ((pa[:, :task_ch] - pb[:, :task_ch]) ** 2).mean()
            else:
                linv = ((pa - pb) ** 2).mean()
            loss = lh + lam_inv * linv
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        print(f"    ep{ep} loss={tot / max(nb, 1):.4f}", flush=True)
    module.eval()
    return module


@torch.no_grad()
def apply_module(module, fine, device, bs=64):
    module.eval()
    outs = []
    for i in range(0, len(fine), bs):
        outs.append(module(fine[i : i + bs].to(device)).cpu())
    return torch.cat(outs, 0)


def sample_cells(fine: torch.Tensor, n_win: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(len(fine), size=min(n_win, len(fine)), replace=False)
    return loc(fine[torch.from_numpy(idx)]).numpy()


def kpca_reconstruct(train_x: np.ndarray, val_fine: torch.Tensor, n_comp: int):
    try:
        from sklearn.decomposition import KernelPCA
    except ImportError:
        return None
    pca = KernelPCA(
        n_components=min(n_comp, len(train_x) - 1),
        kernel="rbf",
        gamma=None,
        fit_inverse_transform=True,
        random_state=0,
    )
    pca.fit(train_x)
    recs = []
    bs = 8192
    x_val = loc(val_fine).numpy()
    for i in range(0, len(x_val), bs):
        recs.append(pca.inverse_transform(pca.transform(x_val[i : i + bs])).astype(np.float32))
        print(f"    kpca {min(i + bs, len(x_val))}/{len(x_val)}", flush=True)
    rec = np.concatenate(recs, 0)
    return torch.from_numpy(rec).view(len(val_fine), 16, 16, 128).permute(0, 3, 1, 2).contiguous()


def _whiten_ops(S_task: torch.Tensor, eps: float = 1e-3):
    c = S_task.size(0)
    I = torch.eye(c, dtype=S_task.dtype)
    tr = float(S_task.trace().clamp_min(1e-6))
    e, U = torch.linalg.eigh(S_task + (eps * tr / c) * I)
    e = e.clamp_min(eps * tr / c)
    T = (U * e.rsqrt()) @ U.T
    Tinv = (U * e.sqrt()) @ U.T
    return T, Tinv


def project_whiten_drop(fine: torch.Tensor, T: torch.Tensor, Tinv: torch.Tensor, V_drop: torch.Tensor) -> torch.Tensor:
    """Drop largest eigen-directions of task-whitened pair covariance, then unwhiten."""
    c, h, w = fine.size(1), fine.size(2), fine.size(3)
    x = loc(fine)
    xw = x @ T
    xd = xw - (xw @ V_drop) @ V_drop.T
    y = xd @ Tinv
    return y.view(fine.size(0), h, w, c).permute(0, 3, 1, 2).contiguous()


def main() -> None:
    parser = add_config_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--teacher-ckpt",
        default="checkpoints/direct_h1s_appear_blob_gr40g/teacher_phase_a_best.pt",
    )
    parser.add_argument("--out", default="reports/cadet_diag/experiment_bank.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    c24 = _load24()
    cfg = load_cfg(args)
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("adapter.variance_head", False)
    cfg.set("training.num_workers", 0)
    device = resolve_device(args.device)
    torch.manual_seed(0)
    np.random.seed(0)
    print(json.dumps(describe_hardware(device), indent=2), flush=True)
    records = load_manifest(cfg.path("paths.manifest"))
    splits = load_splits(cfg.path("paths.split_file"))
    recs = {r.clip_id: r for r in records}
    model = load_frozen_g(cfg, Path(args.teacher_ckpt), device, variance_head=False)
    for p in model.parameters():
        p.requires_grad = False
    caches = {}
    for sp in ("val", "train"):
        print("collect", sp, flush=True)
        ds = TeacherDataset(cfg, records, sp, splits, dummy_backbone=False, use_token_cache=True)
        caches[sp] = c24.collect(model, ds, cfg, device, recs)
    train, val = caches["train"], caches["val"]
    print("fit covariances on TRAIN only", flush=True)
    st = task_pair_stats(train["fine"], train["gt"], train["mate"])
    S_pair, S_task, S_B, S_W = st["S_pair"], st["S_task"], st["S_B"], st["S_W"]
    print(
        json.dumps({k: int(st[k]) for k in ("n_pairs", "n_occ", "n_emp")}),
        flush=True,
    )

    print("GED / Fisher / PLS eigensystems", flush=True)
    lam_ged, U_ged = gevd(S_pair, S_task)  # small λ = low pair / high task
    T_task, Tinv_task = _whiten_ops(S_task)
    A_tw = T_task @ S_pair @ T_task
    A_tw = 0.5 * (A_tw + A_tw.T)
    lam_tw, V_tw = torch.linalg.eigh(A_tw)
    # pair-aware Fisher: S_B v = λ (S_W + S_pair) v, keep large λ
    lam_fi, U_fi = gevd(S_B, S_W + S_pair)
    # PLS: S_B v = λ S_pair v (task vs pair), keep large
    lam_pls, U_pls = gevd(S_B, S_pair + 1e-3 * torch.eye(128) * float(S_pair.trace()) / 128)

    print("task gradients on train subset", flush=True)
    fisher, cov_g = fit_gradients(model.decoder, train["fine"], train["gt"], device, max_n=2048)
    lam_g, U_g = torch.linalg.eigh(0.5 * (cov_g + cov_g.T))  # large = task-sensitive
    lam_fg, U_fg = gevd(S_pair, fisher + 0.05 * torch.eye(128) * float(fisher.trace()) / 128)

    print("spatial residual basis from train Δ", flush=True)
    lo, hi = pair_lo_hi(train["mate"])
    dlt = (train["fine"][lo] - train["fine"][hi]).reshape(len(lo), 128, 256)
    # shared spatial basis: SVD of [128P, 256]
    spat = dlt.permute(0, 1, 2).reshape(-1, 256)
    spat = spat - spat.mean(0)
    # covariance 256x256
    nsp = spat.size(0)
    cov_s = (spat.T @ spat) / max(nsp, 1)
    lam_s, Vs = torch.linalg.eigh(cov_s)  # large last

    pred_g = c24.decode(model, val["fine"], val["z"], device)
    g_rep = warning_report(
        pred_g, val["gt"].numpy(), pre_fork=val["pre"], collision=val["coll"],
        blob_n=val["blob"], paired=val["paired"],
    )
    results = []
    apply_print_decode("G", val["fine"], model, val, device, c24, g_rep, results)

    plans = []
    # 1 GED keep smallest-λ (pair-poor / task-rich)
    for k in (120, 112, 96):
        plans.append((f"1 GED keep {k} low-λ", project_keep(val["fine"], U_ged[:, :k])))
    # 2 task-whitened: drop largest of A in whitened space, then unwhiten
    for k in (8, 16, 32):
        plans.append((
            f"2 task-whitened drop top-{k}",
            project_whiten_drop(val["fine"], T_task, Tinv_task, V_tw[:, -k:]),
        ))
    # 3 pair-aware Fisher keep largest-λ
    for k in (120, 96, 64):
        plans.append((f"3 pair-Fisher keep {k} high-λ", project_keep(val["fine"], U_fi[:, -k:])))
    # 4 task-gradient subspace keep large cov(g)
    for k in (120, 96, 64):
        plans.append((f"4 grad-subspace keep {k}", project_keep(val["fine"], U_g[:, -k:])))
    # 5 Fisher-info GED (S_pair vs empirical Fisher)
    for k in (120, 112, 96):
        plans.append((f"5 Fisher-info GED keep {k} low-λ", project_keep(val["fine"], U_fg[:, :k])))
    # 7 pair-regularized PLS
    for k in (120, 96, 64):
        plans.append((f"7 pair-PLS keep {k} high-λ", project_keep(val["fine"], U_pls[:, -k:])))
    # 8 low-rank spatial residual removal
    for r in (1, 2, 4, 8):
        V = Vs[:, -r:]  # [256, r]
        x = val["fine"].reshape(len(val["fine"]), 128, 256)
        x2 = x - (x @ V) @ V.T
        plans.append((f"8 spatial residual r={r}", x2.view_as(val["fine"])))

    for name, fstar in plans:
        apply_print_decode(name, fstar, model, val, device, c24, g_rep, results)

    # 6 orthogonal rotation (trained)
    print("train 6 orthogonal Q", flush=True)
    oq = OrthoQ().to(device)
    train_linear_map(oq, model.decoder, train, device, epochs=4, lam_inv=0.1, task_ch=96)
    fq = apply_module(oq, val["fine"], device)
    apply_print_decode("6 ortho Q (all 128 ch)", fq, model, val, device, c24, g_rep, results)
    fq_drop = fq.clone()
    fq_drop[:, 96:] = 0
    apply_print_decode("6 ortho Q zero last-32", fq_drop, model, val, device, c24, g_rep, results)

    # 9 Kernel PCA
    print("9 Kernel PCA (train landmarks only)", flush=True)
    rng = np.random.default_rng(0)
    xtr = sample_cells(train["fine"], n_win=16, rng=rng)  # 16 windows × 256 cells
    for nc in (16, 32):
        print(f"  fit KernelPCA n={nc} on {len(xtr)} train cells", flush=True)
        fkp = kpca_reconstruct(xtr, val["fine"], nc)
        if fkp is None:
            print("  sklearn KernelPCA missing, skip", flush=True)
            break
        apply_print_decode(f"9 KernelPCA n={nc} preimage", fkp, model, val, device, c24, g_rep, results)

    # 10 invertible coupling
    print("train 10 invertible coupling", flush=True)
    cpl = Coupling().to(device)
    train_linear_map(cpl, model.decoder, train, device, epochs=4, lam_inv=0.1, task_ch=None)
    fiv = apply_module(cpl, val["fine"], device)
    apply_print_decode("10 invertible coupling", fiv, model, val, device, c24, g_rep, results)

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "fit_split": "train",
        "n_train": len(train["fine"]),
        "n_train_pairs": st["n_pairs"],
        "n_val": len(val["fine"]),
        "ged_eig_smallest5": [float(x) for x in lam_ged[:5]],
        "ged_eig_largest5": [float(x) for x in lam_ged[-5:]],
        "note": "All linear maps fit on train pairs only. Inference is T(F) per clip.",
    }
    path.write_text(json.dumps({"meta": meta, "results": results}, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)
    print("\nRANK by STOP then rec5", flush=True)
    for r in sorted(results, key=lambda x: (x["stop_recall"], x["recall_at_5fa"]), reverse=True):
        print(
            f"  {r['stop_recall']:.3f} rec5={r['recall_at_5fa']:.3f} SAFE={r['safe_recall']:.3f} "
            f"flip={r['warn_flip']:.3f} night={r['night_stop']} dusk={r['dusk_stop']} "
            f"glare={r['glare_stop']}  {r['name']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
