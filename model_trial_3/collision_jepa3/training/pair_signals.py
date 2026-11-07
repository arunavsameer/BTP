"""Matched-background training signals. Inference is unchanged: one clip → D(F)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

DIR_ALPHAS = (0.25, 0.5, 0.75)
DEFAULT_PAIR_GT_TOL = 1e-3


def pair_gt_consistent(
    gt_a: torch.Tensor,
    gt_b: torch.Tensor,
    tol: float = DEFAULT_PAIR_GT_TOL,
) -> torch.Tensor:
    """True where matched backgrounds share the same training ``h_1s`` (max cell |Δ| < tol)."""
    return (gt_a.float() - gt_b.float()).abs().amax(dim=(1, 2)) < float(tol)


def pair_gt_valid_from_numpy(G: np.ndarray, a: np.ndarray, b: np.ndarray, tol: float = DEFAULT_PAIR_GT_TOL) -> np.ndarray:
    """Dense-dump helper: GT agreement on ``gt_1s`` rows for pair indices ``a``, ``b``."""
    diff = np.abs(G[a].astype(np.float64) - G[b].astype(np.float64)).max(axis=(1, 2))
    return diff < float(tol)


def load_mate_h1s(dataset, batch: dict, idx: torch.Tensor) -> torch.Tensor | None:
    """Training labels ``h_1s`` for the mate clip at the same anchors as batch rows ``idx``."""
    if dataset is None or not hasattr(dataset, "h1s_at_anchor"):
        return None
    idx_list = idx.detach().cpu().tolist()
    mates = list(batch.get("mate_clip_id") or [""] * (max(idx_list) + 1 if idx_list else 0))
    anchors = batch["anchor"]
    anchor_list = anchors.detach().cpu().tolist() if torch.is_tensor(anchors) else list(anchors)
    rows: list[np.ndarray] = []
    for i in idx_list:
        mate = mates[i] if i < len(mates) else ""
        if not mate:
            return None
        try:
            rows.append(dataset.h1s_at_anchor(str(mate), int(anchor_list[i])))
        except Exception:
            return None
    if not rows:
        return None
    ref = batch["h_1s"]
    return torch.from_numpy(np.stack(rows, axis=0)).to(device=ref.device, dtype=ref.dtype)


def _subset_pair_valid(
    pair_valid: torch.Tensor,
    *tensors: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if pair_valid.all():
        return (pair_valid, *tensors)
    if not bool(pair_valid.any()):
        empty = []
        for t in tensors:
            if t is None:
                empty.append(None)
            else:
                empty.append(t[:0])
        return (pair_valid, *empty)
    keep = pair_valid.nonzero(as_tuple=True)[0]
    out = []
    for t in tensors:
        if t is None:
            out.append(None)
        else:
            out.append(t.index_select(0, keep))
    return (pair_valid, *out)


def is_canonical_background(background_id: str, clip_id: str) -> bool:
    """Fire pair extras once per pair (bg_00), so interpolants are not doubled."""
    bg = str(background_id)
    cid = str(clip_id)
    return bg.endswith("bg_00") or bg == "00" or "/bg_00/" in cid or cid.endswith("/bg_00")


def subset_batch(batch: dict, idx: torch.Tensor) -> dict:
    n = None
    for v in batch.values():
        if torch.is_tensor(v):
            n = int(v.shape[0])
            break
    if n is None:
        n = len(batch.get("clip_id") or [])
    idx = idx.to(dtype=torch.long)
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and v.dim() > 0 and int(v.shape[0]) == n:
            out[k] = v.index_select(0, idx)
        elif isinstance(v, list) and len(v) == n:
            out[k] = [v[int(i)] for i in idx.tolist()]
        else:
            out[k] = v
    return out


def subset_out(out: dict, idx: torch.Tensor) -> dict:
    n = next(iter(out.values())).shape[0]
    idx = idx.to(dtype=torch.long)
    return {
        k: (v.index_select(0, idx) if torch.is_tensor(v) and v.dim() > 0 and int(v.shape[0]) == n else v)
        for k, v in out.items()
    }


def pair_indices_from_index(idx: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return paired row indices ``a``, ``b`` for two-background keys."""
    key: dict[tuple, dict[str, int]] = {}
    for i, r in enumerate(idx):
        k = (r["sample_id"], r["outcome"], int(r["frame"]))
        key.setdefault(k, {})[str(r["background_id"])] = i
    pairs = [tuple(v.values()) for v in key.values() if len(v) == 2]
    if not pairs:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    a = np.array([p[0] for p in pairs], dtype=np.int64)
    b = np.array([p[1] for p in pairs], dtype=np.int64)
    return a, b


def filter_gt_matched_pairs(
    G: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    tol: float = DEFAULT_PAIR_GT_TOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only pairs whose ``gt_1s`` rows agree within ``tol``."""
    valid = pair_gt_valid_from_numpy(G, a, b, tol=tol)
    return a[valid], b[valid], valid


def load_mate_tokens(dataset, batch: dict, device: str) -> tuple[torch.Tensor | None, list[int]]:
    """Load the matched-background tokens for valid pairs in this batch."""
    toks: list[torch.Tensor] = []
    idx: list[int] = []
    valid = batch["bg_pair_valid"]
    valid_list = valid.detach().cpu().tolist() if torch.is_tensor(valid) else list(valid)
    anchors = batch["anchor"]
    anchor_list = anchors.detach().cpu().tolist() if torch.is_tensor(anchors) else list(anchors)
    mates = list(batch.get("mate_clip_id") or [""] * len(valid_list))
    for i, (ok, mate, t) in enumerate(zip(valid_list, mates, anchor_list)):
        if not bool(ok) or not mate:
            continue
        try:
            arr = dataset._load_tokens(str(mate), int(t))
        except Exception:
            continue
        toks.append(torch.from_numpy(np.asarray(arr, dtype=np.float32)))
        idx.append(i)
    if not toks:
        return None, []
    stacked = torch.stack(toks, 0).to(device=device, non_blocking=True)
    return stacked, idx


def canonical_pair_indices(batch: dict, idx: list[int]) -> list[int]:
    bgs = list(batch.get("background_id") or [""] * (max(idx) + 1 if idx else 0))
    cids = list(batch.get("clip_id") or [""] * (max(idx) + 1 if idx else 0))
    return [i for i in idx if is_canonical_background(bgs[i] if i < len(bgs) else "", cids[i] if i < len(cids) else "")]


def per_sample_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    cell = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    val = cell.mean(dim=(-2, -1))
    if mask is None:
        return val
    keep = ~mask if mask.dtype == torch.bool else mask < 0.5
    return val * keep.float()


def decoder_heatmap_total(loss_fn, out: dict, batch: dict) -> torch.Tensor:
    """G heatmap terms only (no close / L_bg). Used on interpolated F."""
    h_hat = out["Hhat_1s"]
    row_w = loss_fn._row_w(h_hat)
    mask = batch["amb_1s"]
    valid = ~mask if mask.dtype == torch.bool else (mask < 0.5)
    h_tgt = batch["h_1s"]
    logits = out.get("Hhat_1s_logits")
    l_h = loss_fn._map(
        h_hat,
        h_tgt,
        mask=valid,
        row_w=row_w,
        recall_beta=loss_fn.recall_beta,
        cell_weight=loss_fn._cell_w(batch.get("h_now"), h_tgt),
        logits=logits,
    )
    l_blob = loss_fn._blob(h_hat, h_tgt, valid, logits=logits)
    l_stop = loss_fn._stop_hinge(h_hat, h_tgt, valid, logits=logits)
    l_safe = loss_fn._safe_hinge(h_hat, h_tgt, valid, logits=logits)
    from ..losses.heatmap import false_positive_penalty
    from ..losses.direction import direction_alignment_loss

    l_fp = false_positive_penalty(
        h_hat, h_tgt, sample_mask=valid, safe_threshold=loss_fn.safe_thr, margin=loss_fn.fp_margin
    )
    l_dir = direction_alignment_loss(
        h_hat,
        h_tgt,
        row_weights=row_w,
        left_cols=loss_fn.left,
        center_cols=loss_fn.center,
        right_cols=loss_fn.right,
        sample_mask=valid,
        temperature=loss_fn.dir_temp,
        threat_threshold=loss_fn.dir_thr,
    )
    return (
        l_h
        + loss_fn.blob_w * l_blob
        + loss_fn.stop_hinge_w * l_stop
        + loss_fn.safe_hinge_w * l_safe
        + loss_fn.fp_w * l_fp
        + loss_fn.dir_w * l_dir
    )


def pair_dir_adv_loss(
    model,
    loss_fn,
    *,
    z_a: torch.Tensor,
    fine_a: torch.Tensor,
    fine_b: torch.Tensor,
    batch_a: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Hardest interpolant along ΔF = Fb−Fa. α* has no gradient."""
    d_f = fine_b - fine_a
    mask = batch_a["amb_1s"]
    y = batch_a["h_1s"]
    with torch.no_grad():
        scores = []
        for alpha in DIR_ALPHAS:
            out_a = model._heads(z_a, fine_a + float(alpha) * d_f)
            scores.append(per_sample_bce(out_a["Hhat_1s_logits"], y, mask))
        stacked = torch.stack(scores, 0)
        pick = stacked.argmax(dim=0)
    alphas = z_a.new_tensor(DIR_ALPHAS)
    a_star = alphas[pick].view(-1, 1, 1, 1)
    f_hard = fine_a + a_star * d_f
    out_h = model._heads(z_a, f_hard)
    loss = decoder_heatmap_total(loss_fn, out_h, batch_a)
    counts = {f"dir_adv_alpha_{a}": 0.0 for a in DIR_ALPHAS}
    for p in pick.detach().cpu().tolist():
        counts[f"dir_adv_alpha_{DIR_ALPHAS[int(p)]}"] += 1.0
    n = max(len(pick), 1)
    stats = {"L_dir_adv": float(loss.detach()), "dir_adv_n": float(len(pick))}
    stats.update({k: v / n for k, v in counts.items()})
    return loss, stats


def pcgrad_combine(
    ga: list[torch.Tensor],
    gb: list[torch.Tensor],
) -> tuple[list[torch.Tensor], float, bool]:
    """PCGrad: if ga·gb < 0, project each off the other, then average."""
    flat_a = torch.cat([g.reshape(-1) for g in ga])
    flat_b = torch.cat([g.reshape(-1) for g in gb])
    denom = (flat_a.norm() * flat_b.norm()).clamp_min(1e-12)
    cos = float((flat_a @ flat_b) / denom)
    conflict = cos < 0.0
    if not conflict:
        mixed = [(a + b) * 0.5 for a, b in zip(ga, gb)]
        return mixed, cos, False
    a2 = (flat_a @ flat_a).clamp_min(1e-12)
    b2 = (flat_b @ flat_b).clamp_min(1e-12)
    dot = flat_a @ flat_b
    ga_t = [a - (dot / b2) * b for a, b in zip(ga, gb)]
    gb_t = [b - (dot / a2) * a for a, b in zip(ga, gb)]
    mixed = [(a + b) * 0.5 for a, b in zip(ga_t, gb_t)]
    return mixed, cos, True


def _zeros_like_unused(grads, params):
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


def pcgrad_grads(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    loss_u: torch.Tensor | None,
    params: list[torch.nn.Parameter],
) -> tuple[list[torch.Tensor], dict[str, float]]:
    need_graph = loss_u is not None and float(loss_u.detach().abs()) > 0
    ga = torch.autograd.grad(loss_a, params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss_b, params, retain_graph=need_graph, allow_unused=True)
    ga = _zeros_like_unused(ga, params)
    gb = _zeros_like_unused(gb, params)
    mixed, cos, conflict = pcgrad_combine(ga, gb)
    if need_graph:
        gu = torch.autograd.grad(loss_u, params, retain_graph=False, allow_unused=True)
        gu = _zeros_like_unused(gu, params)
        grads = [u + m for u, m in zip(gu, mixed)]
    else:
        grads = mixed
    stats = {
        "pcgrad_cos": cos,
        "pcgrad_conflict": 1.0 if conflict else 0.0,
        "L_pcgrad_a": float(loss_a.detach()),
        "L_pcgrad_b": float(loss_b.detach()),
    }
    return grads, stats


def encode_batch(model, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if "tokens" in batch:
        return model.encode_pair(tokens=batch["tokens"])
    return model.encode_pair(clip=batch["frames"])


def occupancy_balanced_huber(
    za: torch.Tensor,
    zb: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    eps: float = 1e-6,
    beta: float = 1.0,
) -> torch.Tensor:
    """Occupancy-balanced Huber on 5×5 logits. Empty GT maps use the empty term only."""
    ell = F.smooth_l1_loss(za.float(), zb.float(), reduction="none", beta=beta)
    pos = y.float().clamp(0.0, 1.0)
    neg = 1.0 - pos
    if mask is not None:
        sm = (~mask if mask.dtype == torch.bool else mask < 0.5).to(dtype=ell.dtype, device=ell.device)
        sm = sm.view(-1, *([1] * (ell.dim() - 1)))
        pos = pos * sm
        neg = neg * sm
        if float(sm.sum()) <= 0:
            return za.new_zeros(())
    l_neg = (ell * neg).sum() / neg.sum().clamp(min=eps)
    if float(pos.sum()) <= eps:
        return l_neg
    l_pos = (ell * pos).sum() / pos.sum().clamp(min=eps)
    return 0.5 * (l_pos + l_neg)


def masked_huber_mean(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    beta: float = 1.0,
) -> torch.Tensor:
    ell = F.smooth_l1_loss(a.float(), b.float(), reduction="none", beta=beta)
    val = ell.reshape(ell.shape[0], -1).mean(dim=1)
    if mask is None:
        return val.mean()
    keep = ~mask if mask.dtype == torch.bool else mask < 0.5
    keep = keep.to(dtype=val.dtype, device=val.device).view(-1)
    if float(keep.sum()) <= 0:
        return a.new_zeros(())
    return (val * keep).sum() / keep.sum()


def bounded_pair_perturbation(
    fa: torch.Tensor,
    fb: torch.Tensor,
    *,
    rho_lo: float = 0.05,
    rho_hi: float = 0.10,
    rms_frac: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fa' = Fa + ρΔF, Fb' = Fb − ρΔF, with ΔF = sg(Fb−Fa) and RMS(ρΔF) ≤ rms_frac · RMS(F)."""
    delta = (fb - fa).detach()
    bsz = int(fa.shape[0])
    rho = torch.empty(bsz, device=fa.device, dtype=fa.dtype).uniform_(float(rho_lo), float(rho_hi))
    view = [bsz] + [1] * (fa.dim() - 1)
    pert = rho.view(*view) * delta
    rms_p = pert.flatten(1).pow(2).mean(dim=1).sqrt().clamp(min=1e-8)
    rms_f = 0.5 * (
        fa.detach().flatten(1).pow(2).mean(dim=1).sqrt()
        + fb.detach().flatten(1).pow(2).mean(dim=1).sqrt()
    )
    cap = float(rms_frac) * rms_f
    scale = (cap / rms_p).clamp(max=1.0)
    pert = pert * scale.view(*view)
    return fa + pert, fb - pert, rho, scale


def pair_padr_loss(
    model,
    *,
    z_a: torch.Tensor,
    fine_a: torch.Tensor,
    z_b: torch.Tensor,
    fine_b: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None,
    za: torch.Tensor | None = None,
    pair_valid: torch.Tensor | None = None,
    rho_lo: float = 0.05,
    rho_hi: float = 0.10,
    rms_frac: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Actual-pair logit agreement plus local sensitivity along sg(Fb−Fa)."""
    n_in = int(y.shape[0])
    if pair_valid is not None:
        n_valid = int(pair_valid.sum())
        n_skip = n_in - n_valid
    else:
        n_valid = n_in
        n_skip = 0
    if pair_valid is not None and n_valid == 0:
        z = y.new_zeros(())
        return z, z, {
            "L_pair": 0.0,
            "L_tangent": 0.0,
            "padr_n": float(n_in),
            "padr_n_gt_valid": 0.0,
            "padr_n_gt_skip": float(n_skip),
        }

    if za is None:
        za = model._heads(z_a, fine_a)["Hhat_1s_logits"]
    zb = model._heads(z_b, fine_b)["Hhat_1s_logits"]
    if pair_valid is not None and not pair_valid.all():
        _, za, zb, z_a, fine_a, z_b, fine_b, y, mask = _subset_pair_valid(
            pair_valid, za, zb, z_a, fine_a, z_b, fine_b, y, mask
        )
        if za is None or int(za.shape[0]) == 0:
            z = y.new_zeros(())
            return z, z, {
                "L_pair": 0.0,
                "L_tangent": 0.0,
                "padr_n": float(n_in),
                "padr_n_gt_valid": 0.0,
                "padr_n_gt_skip": float(n_skip),
            }

    l_pair = occupancy_balanced_huber(za, zb, y, mask)
    fa_p, fb_p, rho, scale = bounded_pair_perturbation(
        fine_a, fine_b, rho_lo=rho_lo, rho_hi=rho_hi, rms_frac=rms_frac
    )
    za_p = model._heads(z_a, fa_p)["Hhat_1s_logits"]
    zb_p = model._heads(z_b, fb_p)["Hhat_1s_logits"]
    l_tan = 0.5 * (
        masked_huber_mean(za_p, za.detach(), mask) + masked_huber_mean(zb_p, zb.detach(), mask)
    )
    stats = {
        "L_pair": float(l_pair.detach()),
        "L_tangent": float(l_tan.detach()),
        "padr_n": float(n_in),
        "padr_n_gt_valid": float(n_valid),
        "padr_n_gt_skip": float(n_skip),
        "padr_rho": float(rho.mean().detach()),
        "padr_cap_scale": float(scale.mean().detach()),
    }
    return l_pair, l_tan, stats
