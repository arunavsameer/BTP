"""Background-pair regularization: token-channel dropout and output consistency loss."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def load_channel_sets_from_p4(p4_json: Path) -> tuple[list[int], int, list[int]]:
    """Return (targeted channel ids, n_eligible, all eligible ids for random control).

    Eligible = channels with W >= median(W) from ``channel_icc.npz`` next to ``p4_json``.
    """
    p4_json = Path(p4_json)
    rep = json.loads(p4_json.read_text(encoding="utf-8"))
    targeted = [int(x) for x in rep["selection"]["selected_channel_ids"]]
    n_eligible = int(rep["selection"]["n_eligible"])
    icc_dir = p4_json.parent
    w_path = icc_dir / "channel_icc.npz"
    if not w_path.exists():
        return targeted, n_eligible, list(range(1408))
    W = np.load(w_path)["W"].astype(np.float64)
    med = float(np.median(W))
    eligible_ids = [int(i) for i in np.where(W >= med)[0]]
    if len(eligible_ids) != n_eligible:
        eligible_ids = [int(i) for i in range(1408) if float(W[i]) >= med]
    return targeted, n_eligible, eligible_ids


def train_h_weight_mean(cfg, train_ds) -> float:
    """Global mean of w_ij = 0.05 + 4*Y_ij^2 over training h_1s (valid cells)."""
    acc = 0.0
    n = 0
    for i in range(len(train_ds)):
        s = train_ds[i]
        y = np.asarray(s["h_1s"], dtype=np.float64).reshape(-1)
        amb = s["amb_1s"]
        if isinstance(amb, np.ndarray) and amb.dtype == bool:
            m = ~amb.reshape(-1)
        else:
            m = np.ones(y.shape[0], dtype=bool)
        w = 0.05 + 4.0 * y * y
        acc += float(w[m].sum())
        n += int(m.sum())
    return max(acc / max(n, 1), 1e-8)


def pair_consistency_loss(
    ha: torch.Tensor,
    hb: torch.Tensor,
    y_ref: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    w_train_mean: float,
) -> torch.Tensor:
    """Mean of tilde{w} * (Ha - Hb)^2 on 1s heatmaps (sigmoid space)."""
    diff = ha - hb
    sq = diff * diff
    y = y_ref.float()
    w = 0.05 + 4.0 * y * y
    tw = w / float(w_train_mean)
    if mask is not None:
        sm = (~mask if mask.dtype == torch.bool else (mask < 0.5)).to(dtype=sq.dtype, device=sq.device)
        sm = sm.view(ha.shape[0], 1, 1).expand_as(sq)
        num = (tw * sq * sm).sum()
        den = sm.sum().clamp(min=1.0)
        return num / den
    return (tw * sq).mean()


def token_channel_dropout(
    tokens: torch.Tensor,
    channel_ids: Sequence[int],
    *,
    training: bool,
    p_drop: float = 0.5,
) -> torch.Tensor:
    """Inverted dropout on last dim (hidden) for indices in ``channel_ids``.

    Keep prob = 1 - p_drop; scale by 1/(1-p_drop) when kept (PyTorch convention).
    ``tokens`` shape [N, T, C] or [N, C] — mask broadcasts on leading dims.
    """
    if not training or not channel_ids:
        return tokens
    device = tokens.device
    dtype = tokens.dtype
    c_dim = int(tokens.shape[-1])
    p_keep = max(1.0 - float(p_drop), 1e-8)
    mask = torch.ones(c_dim, device=device, dtype=dtype)
    cid = torch.tensor(list(channel_ids), device=device, dtype=torch.long)
    keep = (torch.rand(len(cid), device=device, dtype=torch.float32) < p_keep).to(dtype=dtype)
    mask[cid] = keep / p_keep
    if tokens.dim() == 3:
        return tokens * mask.view(1, 1, -1)
    return tokens * mask.view(1, -1)


def shared_pair_dropout_mask(
    channel_ids: Sequence[int],
    *,
    c_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    p_drop: float = 0.5,
) -> torch.Tensor:
    """One mask [C] shared by both clips of a pair."""
    p_keep = max(1.0 - float(p_drop), 1e-8)
    mask = torch.ones(c_dim, device=device, dtype=dtype)
    if not channel_ids:
        return mask
    cid = torch.tensor(list(channel_ids), device=device, dtype=torch.long)
    keep = (torch.rand(len(cid), device=device, dtype=torch.float32) < p_keep).to(dtype=dtype)
    mask[cid] = keep / p_keep
    return mask


def apply_token_mask(tokens: torch.Tensor, mask_c: torch.Tensor) -> torch.Tensor:
    if tokens.dim() == 3:
        return tokens * mask_c.view(1, 1, -1)
    return tokens * mask_c.view(1, -1)


def bg_reg_pair_strength(epoch: int) -> float:
    """0 in ep 0–2, then 0.25, 0.5, 0.75, 1.0 from ep 3 onward."""
    if epoch < 3:
        return 0.0
    if epoch == 3:
        return 0.25
    if epoch == 4:
        return 0.50
    if epoch == 5:
        return 0.75
    return 1.0


def bg_reg_dropout_active(epoch: int, *, dropout_on: bool) -> bool:
    if not dropout_on:
        return False
    return epoch >= 3


def clean_anchor_drop_strength(epoch: int) -> float:
    """Ramp auxiliary dropout path: ep0 off, then 0.25, 0.5, 0.75, then 1.0."""
    e = int(epoch)
    if e <= 0:
        return 0.0
    if e == 1:
        return 0.25
    if e == 2:
        return 0.50
    if e == 3:
        return 0.75
    return 1.0


pair_cf_ramp_strength = clean_anchor_drop_strength


def apply_pair_counterfactual_shrink(
    tokens: torch.Tensor,
    U: torch.Tensor,
    *,
    lam: float = 1.0,
) -> torch.Tensor:
    """Pooled D PCA shrinkage broadcast across tokens. U [k,C]; tokens [B,T,C] paired layout."""
    if tokens.shape[0] < 2 or tokens.shape[0] % 2 != 0:
        return tokens
    U = U.to(device=tokens.device, dtype=torch.float32)
    mid = tokens.shape[0] // 2
    Fa = tokens[:mid]
    Fb = tokens[mid:]
    mean_a = Fa.float().mean(dim=1)
    mean_b = Fb.float().mean(dim=1)
    D = 0.5 * (mean_a - mean_b)
    d = float(lam) * ((D @ U.T) @ U)
    out = tokens.clone()
    out[:mid] = Fa.to(dtype=tokens.dtype) - d.unsqueeze(1).to(dtype=tokens.dtype)
    out[mid:] = Fb.to(dtype=tokens.dtype) + d.unsqueeze(1).to(dtype=tokens.dtype)
    return out


def calibrate_beta_pair_counterfactual(
    cfg,
    model,
    loss_fn,
    train_loader,
    device: str,
    U: torch.Tensor,
    *,
    calib_epoch: int = 1,
    n_batches: int = 100,
    eps: float = 1e-8,
) -> float:
    """β = grad_scale / median(||g_cf|| / (||g_clean|| + ε))."""
    from .engine import move_batch, teacher_forward

    bg = cfg.get("training.bg_reg") or {}
    lam = float(bg.get("pair_cf_lambda", 1.0))
    frac = float(bg.get("pair_cf_pair_frac", 0.5))
    scale = float(bg.get("pair_cf_grad_scale", 0.10))
    ratios: list[float] = []
    model.train()
    if hasattr(model, "backbone"):
        model.backbone.eval()
    n_done = 0
    U = U.to(device=device, dtype=torch.float32)
    for batch in train_loader:
        batch = move_batch(batch, device)
        if "tokens" not in batch or not bool(batch.get("bg_reg_paired", False)):
            continue
        bsz = int(batch["tokens"].shape[0])
        if bsz < 2 or bsz % 2 != 0:
            continue
        pair_ids = batch.get("background_pair_id") or []
        pid = str(pair_ids[0]) if pair_ids else ""
        key_pick = f"{int(calib_epoch)}|{pid}|pair_cf_aux"
        h = hashlib.sha256(key_pick.encode("utf-8")).digest()
        thr = min(255, max(0, int(round(frac * 256))))
        if not (h[0] < thr):
            continue
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out_c = teacher_forward(model, batch)
            l_clean = loss_fn(out_c, batch).total
        g_clean = adapter_trainable_grad_norm(l_clean, model)
        model.zero_grad(set_to_none=True)
        tok_cf = apply_pair_counterfactual_shrink(batch["tokens"], U, lam=lam)
        b_cf = {**batch, "tokens": tok_cf}
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out_cf = teacher_forward(model, b_cf)
            l_cf = loss_fn(out_cf, b_cf).total
        g_cf = adapter_trainable_grad_norm(l_cf, model)
        model.zero_grad(set_to_none=True)
        if g_clean > eps:
            ratios.append(float(g_cf / (g_clean + eps)))
        n_done += 1
        if n_done >= n_batches:
            break
    if not ratios:
        return scale * 1.0
    med = float(np.median(np.asarray(ratios, dtype=np.float64)))
    return scale / max(med, eps)


def deterministic_channel_mask(
    channel_ids: Sequence[int],
    *,
    c_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    p_drop: float,
    key: str,
) -> torch.Tensor:
    """Inverted dropout mask [C] deterministic from UTF-8 ``key`` (shared across pair halves)."""
    p_keep = max(1.0 - float(p_drop), 1e-8)
    mask = torch.ones(c_dim, device=device, dtype=dtype)
    if not channel_ids:
        return mask
    cid = torch.tensor(list(channel_ids), device=device, dtype=torch.long)
    h = hashlib.sha256(key.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "little") & 0xFFFFFFFFFFFFFFFF
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    u = torch.rand(len(cid), generator=gen, device=device, dtype=torch.float32)
    keep = (u < p_keep).to(dtype=dtype)
    mask[cid] = keep / p_keep
    return mask


def adapter_trainable_grad_norm(loss: torch.Tensor, model) -> float:
    """L2 norm of gradients w.r.t. adapter+decoder parameters only."""
    params = []
    for p in list(model.adapter.parameters()) + list(model.decoder.parameters()):
        if p.requires_grad:
            params.append(p)
    if not params:
        return 0.0
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().float().pow(2).sum())
    return math.sqrt(total)


def adapter_grad_norm(loss: torch.Tensor, model) -> float:
    params = model.adapter_parameters()
    grads = torch.autograd.grad(loss, params, retain_graph=True, create_graph=False, allow_unused=True)
    total = 0.0
    for g in grads:
        if g is not None:
            total += float(g.detach().float().pow(2).sum())
    return math.sqrt(total)


def calibrate_beta_clean_anchor(
    cfg,
    model,
    loss_fn,
    train_loader,
    device: str,
    channel_ids: Sequence[int],
    *,
    calib_epoch: int = 1,
    n_batches: int = 50,
    eps: float = 1e-8,
) -> float:
    """β = clean_anchor_grad_scale * median(||g_clean|| / (||g_drop|| + ε)) over up to ``n_batches``."""
    from .engine import move_batch, teacher_forward

    bg = cfg.get("training.bg_reg") or {}
    p_drop = float(bg.get("dropout_prob", 0.25))
    frac = float(bg.get("clean_anchor_pair_frac", 0.5))
    scale = float(bg.get("clean_anchor_grad_scale", 0.10))
    ratios: list[float] = []
    model.train()
    if hasattr(model, "backbone"):
        model.backbone.eval()
    n_done = 0
    for batch in train_loader:
        batch = move_batch(batch, device)
        if "tokens" not in batch:
            continue
        if not bool(batch.get("bg_reg_paired", False)):
            continue
        tok = batch["tokens"]
        bsz = int(tok.shape[0])
        if bsz < 2 or bsz % 2 != 0:
            continue
        pair_ids = batch.get("background_pair_id") or []
        pid = str(pair_ids[0]) if pair_ids else ""
        key_pick = f"{int(calib_epoch)}|{pid}|clean_anchor_aux"
        h = hashlib.sha256(key_pick.encode("utf-8")).digest()
        thr = min(255, max(0, int(round(frac * 256))))
        if not (h[0] < thr):
            continue
        mid = bsz // 2
        c_dim = int(tok.shape[-1])
        device_t = tok.device
        dtype = tok.dtype
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out_c = teacher_forward(model, batch)
            packed_c = loss_fn(out_c, batch)
            l_clean = packed_c.total
        g_clean = adapter_trainable_grad_norm(l_clean, model)
        model.zero_grad(set_to_none=True)
        tok_d = tok.clone()
        anchors = batch["anchor"]
        for i in range(mid):
            anc = int(anchors[i].item()) if torch.is_tensor(anchors[i]) else int(anchors[i])
            mk = f"{int(calib_epoch)}|{pair_ids[i]}|{anc}"
            mask_c = deterministic_channel_mask(
                channel_ids, c_dim=c_dim, device=device_t, dtype=dtype, p_drop=p_drop, key=mk
            )
            tok_d[i] = apply_token_mask(tok[i], mask_c)
            tok_d[i + mid] = apply_token_mask(tok[i + mid], mask_c)
        drop_batch = {**batch, "tokens": tok_d}
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out_d = teacher_forward(model, drop_batch)
            l_drop = loss_fn(out_d, drop_batch).total
        g_drop = adapter_trainable_grad_norm(l_drop, model)
        model.zero_grad(set_to_none=True)
        if g_drop > eps:
            ratios.append(float(g_clean / (g_drop + eps)))
        n_done += 1
        if n_done >= n_batches:
            break
    if not ratios:
        return scale * 1.0
    return scale * float(np.median(np.array(ratios, dtype=np.float64)))


def calibrate_lambda_pair(
    cfg,
    model,
    loss_fn,
    train_loader,
    device: str,
    *,
    w_train_mean: float,
    n_batches: int = 50,
    eps: float = 1e-8,
) -> float:
    """``lambda = 0.075 * median(g_task / (g_pair + eps))`` over paired minibatches."""
    from .engine import move_batch
    from .pair_signals import encode_batch, subset_batch, subset_out

    ratios: list[float] = []
    it = iter(train_loader)
    for _ in range(min(n_batches, len(train_loader))):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = move_batch(batch, device)
        if "tokens" not in batch or not bool(batch.get("bg_reg_paired", False)):
            continue
        b = int(batch["tokens"].shape[0])
        if b < 2 or b % 2 != 0:
            continue
        mid = b // 2
        idx_a = torch.arange(0, mid, device=device, dtype=torch.long)
        idx_b = torch.arange(mid, b, device=device, dtype=torch.long)
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            z, fine = encode_batch(model, batch)
            out = model._heads(z, fine)
            pa = loss_fn(subset_out(out, idx_a), subset_batch(batch, idx_a))
            pb = loss_fn(subset_out(out, idx_b), subset_batch(batch, idx_b))
            l_task = 0.5 * (pa.total + pb.total)
        g_t = adapter_grad_norm(l_task, model)
        model.zero_grad(set_to_none=True)

        ha = out["Hhat_1s"][:mid]
        hb = out["Hhat_1s"][mid:]
        pv = batch["bg_pair_gt_valid"][:mid]
        if not bool(pv.any()):
            continue
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            l_p = pair_consistency_loss(
                ha[pv],
                hb[pv],
                batch["h_1s"][:mid][pv],
                batch["amb_1s"][:mid][pv],
                w_train_mean=w_train_mean,
            )
        if float(l_p.detach()) <= 0:
            continue
        g_p = adapter_grad_norm(l_p, model)
        model.zero_grad(set_to_none=True)
        if g_p > eps:
            ratios.append(float(g_t / (g_p + eps)))
    if not ratios:
        return float(cfg.get("training.bg_reg.lambda_pair_default", 0.05))
    return 0.075 * float(np.median(np.array(ratios, dtype=np.float64)))
