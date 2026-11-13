"""Task-whitened pair-covariance residual on 128-ch F. Frozen linear map, single-clip T(F)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training.engine import move_batch


def _pair_key(cid: str) -> tuple[str, str]:
    parts = str(cid).split("/")
    if len(parts) >= 3:
        return f"{parts[0]}|{parts[1]}", parts[2]
    return str(cid), "?"


def loc(fine: torch.Tensor) -> torch.Tensor:
    return fine.permute(0, 2, 3, 1).reshape(-1, fine.size(1))


def cov_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(x.size(0), 1)
    mu = x.mean(0)
    xc = x - mu
    return mu, (xc.T @ xc) / n


def _whiten_ops(S_task: torch.Tensor, eps: float = 1e-3):
    c = S_task.size(0)
    eye = torch.eye(c, dtype=S_task.dtype, device=S_task.device)
    tr = float(S_task.trace().clamp_min(1e-6))
    e, U = torch.linalg.eigh(S_task + (eps * tr / c) * eye)
    e = e.clamp_min(eps * tr / c)
    t_mat = (U * e.rsqrt()) @ U.T
    t_inv = (U * e.sqrt()) @ U.T
    return t_mat, t_inv


def drop_matrix(T: torch.Tensor, Tinv: torch.Tensor, V_drop: torch.Tensor) -> torch.Tensor:
    """W such that x' = x @ W, with x row-vectors [N,C]."""
    c = T.size(0)
    eye = torch.eye(c, dtype=T.dtype, device=T.device)
    return T @ (eye - V_drop @ V_drop.T) @ Tinv


def fit_drop_matrix(fine: torch.Tensor, gt: torch.Tensor, mate, drop_k: int = 16) -> dict[str, Any]:
    """Train-only S_task-whitened S_pair. Drop the largest whitened pair directions."""
    mate = torch.as_tensor(mate)
    i = torch.arange(len(mate))
    ok = (mate >= 0) & (i < mate)
    lo, hi = i[ok], mate[ok]
    d = loc(fine[lo] - fine[hi])
    _, s_pair = cov_xy(d)
    y16 = F.interpolate(gt.unsqueeze(1), size=fine.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    x = loc(fine)
    y = y16.reshape(-1)
    occ = y >= 0.5
    _, s_occ = cov_xy(x[occ] if int(occ.sum()) > 10 else x)
    _, s_all = cov_xy(x)
    mu = x.mean(0)
    s_b = torch.zeros(x.size(1), x.size(1), dtype=x.dtype)
    n_used = 0
    for a, b in zip((0.0, 0.15, 0.40, 0.70), (0.15, 0.40, 0.70, 1.01)):
        m = (y >= a) & (y < b)
        n_k = int(m.sum())
        if n_k < 20:
            continue
        xk = x[m]
        muk = xk.mean(0)
        s_b = s_b + n_k * torch.outer(muk - mu, muk - mu)
        n_used += n_k
    s_b = s_b / max(n_used, 1)
    s_task = s_b + 0.25 * s_occ + 0.05 * s_all
    t_mat, t_inv = _whiten_ops(s_task)
    a_tw = t_mat @ s_pair @ t_mat
    a_tw = 0.5 * (a_tw + a_tw.T)
    lam, v_tw = torch.linalg.eigh(a_tw)
    k = min(int(drop_k), v_tw.size(1))
    v_drop = v_tw[:, -k:]
    w = drop_matrix(t_mat, t_inv, v_drop)
    return {
        "W": w.contiguous(),
        "lam_largest": [float(x) for x in lam[-k:]],
        "n_pairs": int(len(lo)),
        "drop_k": k,
    }


class TaskWhitenDrop(nn.Module):
    """F' = W F on channels. Identity and off until fitted or loaded."""

    def __init__(self, channels: int = 128, drop_k: int = 16):
        super().__init__()
        self.drop_k = int(drop_k)
        self.register_buffer("W", torch.eye(int(channels)))
        self.register_buffer("enabled", torch.zeros((), dtype=torch.uint8))

    def load_W(self, w: torch.Tensor) -> None:
        if tuple(w.shape) != tuple(self.W.shape):
            raise ValueError(f"W shape {tuple(w.shape)} != {tuple(self.W.shape)}")
        self.W.copy_(w.to(device=self.W.device, dtype=self.W.dtype))
        self.enabled.fill_(1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if int(self.enabled.item()) == 0:
            return feat
        return torch.einsum("bchw,cd->bdhw", feat, self.W)


@torch.no_grad()
def collect_train_pack(model, dataset, cfg, device: str) -> dict[str, torch.Tensor]:
    from ..data.dataset import build_dataloader

    loader = build_dataloader(dataset, cfg, shuffle=False)
    model.eval()
    fines, gts, clip_id, anchor = [], [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        if "tokens" in batch:
            _z, fine = model.encode_pair(tokens=batch["tokens"])
        else:
            _z, fine = model.encode_pair(clip=batch["frames"])
        fines.append(fine.float().cpu())
        gts.append(batch["h_1s"].float().cpu())
        clip_id.extend(list(batch["clip_id"]))
        anchor.append(batch["anchor"].cpu())
        if len(clip_id) % 1280 == 0:
            print(f"  task-whiten collect {len(clip_id)}/{len(dataset)}", flush=True)
    fine = torch.cat(fines, 0)
    gt = torch.cat(gts, 0)
    anc = torch.cat(anchor, 0).numpy().astype("int32")
    n = len(clip_id)
    mate = torch.full((n,), -1, dtype=torch.int32)
    buckets: dict[tuple[str, int], dict[str, int]] = defaultdict(dict)
    for i, cid in enumerate(clip_id):
        pid, bg = _pair_key(cid)
        buckets[(pid, int(anc[i]))][bg] = i
    n_pairs = 0
    for bgs in buckets.values():
        if "bg_00" in bgs and "bg_01" in bgs:
            i0, i1 = bgs["bg_00"], bgs["bg_01"]
            mate[i0] = i1
            mate[i1] = i0
            n_pairs += 1
    print(f"  task-whiten fine {tuple(fine.shape)} pairs {n_pairs}", flush=True)
    return {"fine": fine, "gt": gt, "mate": mate, "n_pairs": n_pairs}


@torch.no_grad()
def fit_and_install(model, dataset, cfg, device: str, drop_k: int = 16) -> dict[str, Any]:
    """Fit W on current adapter features (T disabled), then install and enable it."""
    tw = model.task_whiten
    tw.enabled.fill_(0)
    pack = collect_train_pack(model, dataset, cfg, device)
    stats = fit_drop_matrix(pack["fine"], pack["gt"], pack["mate"], drop_k=drop_k)
    tw.load_W(stats["W"])
    print(
        f"  installed task-whiten drop-{stats['drop_k']} n_pairs={stats['n_pairs']} "
        f"lam={stats['lam_largest']}",
        flush=True,
    )
    return {k: v for k, v in stats.items() if k != "W"} | {"n_train": int(pack["fine"].size(0))}
