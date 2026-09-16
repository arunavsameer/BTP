"""Loss functions.

Student objective (see ``student_loss``):

    L = L(H+) + 0.5 L(H_t) + balanced readout + balanced STOP + optional JEPA

Auxiliary terms are rescaled to a *fraction of L(H+)* with a boost cap, so a
large readout/STOP/JEPA number cannot drown the heatmap. Heatmap regression
downweights H≈0 (leaves / empty cells) and ignores already-safe cells.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def weighted_focal_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float = 1.0,
    gamma: float = 2.0,
    change: torch.Tensor | None = None,
    change_weight: float = 0.0,
) -> torch.Tensor:
    """Per-cell MSE upweighted where the *true* threat is high.

    weight = 1 + focal_weight * target^gamma  (target in [0, 1])

    Optional ``change`` (e.g. |H_future - H_now|) further upweights cells that
    actually move, so the model cannot hide in the static corridor prior.
    """
    se = (pred - target) ** 2
    weight = 1.0 + focal_weight * torch.clamp(target, 0.0, 1.0) ** gamma
    if change is not None and change_weight > 0:
        weight = weight + change_weight * torch.clamp(change, 0.0, 1.0)
    return (se * weight).mean()


def jepa_distance(z_hat: torch.Tensor, z: torch.Tensor, cos_weight: float = 1.0) -> torch.Tensor:
    """Distance between predicted and teacher future collision states.

    Teacher Z is unconstrained (observed range ~[-60, 60]). Raw MSE would drown
    the heatmap loss, so both tensors are scaled by the teacher's per-sample std
    before MSE. Cosine is scale-invariant and stays on the raw vectors.
    z, z_hat: [B, z_channels, grid, grid].
    """
    # Per-sample std over (C, grid, grid); use the teacher scale for both sides.
    scale = z.detach().flatten(1).std(dim=1, keepdim=True).clamp_min(1e-3)
    scale = scale.view(-1, 1, 1, 1)
    mse = F.mse_loss(z_hat / scale, z / scale)
    cos = F.cosine_similarity(z_hat, z, dim=1)  # [B, grid, grid]
    cos_term = (1.0 - cos).mean()
    return mse + cos_weight * cos_term


def heatmap_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float = 4.0,
    gamma: float = 2.0,
    change: torch.Tensor | None = None,
    change_weight: float = 0.0,
    beta: float = 0.1,
    bg_weight: float = 0.05,
    ignore_below: float = 0.15,
    fa_weight: float = 1.0,
    fa_pred_thr: float = 0.45,
    fa_true_thr: float = 0.2,
) -> torch.Tensor:
    """Clutter-aware SmoothL1: ignore already-safe cells, downweight H≈0, punish false alarms."""
    return collision_heatmap_loss(
        pred,
        target,
        focal_weight=focal_weight,
        gamma=gamma,
        change=change,
        change_weight=change_weight,
        beta=beta,
        bg_weight=bg_weight,
        ignore_below=ignore_below,
        fa_weight=fa_weight,
        fa_pred_thr=fa_pred_thr,
        fa_true_thr=fa_true_thr,
    )


def collision_heatmap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float = 4.0,
    gamma: float = 2.0,
    change: torch.Tensor | None = None,
    change_weight: float = 0.0,
    beta: float = 0.1,
    bg_weight: float = 0.05,
    ignore_below: float = 0.15,
    fa_weight: float = 1.0,
    fa_pred_thr: float = 0.45,
    fa_true_thr: float = 0.2,
) -> torch.Tensor:
    """Regression that does not spend capacity matching leaf/empty cells.

    * ``w = bg_weight + focal_weight * H^gamma`` so H≈0 is cheap.
    * If both pred and target are below ``ignore_below``, the cell is dropped.
    * Extra penalty when pred exceeds CAUTION on a truly safe cell (false alarm).
    """
    elem = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    safe = (pred < ignore_below) & (target < ignore_below)
    elem = elem * (~safe).to(elem.dtype)
    weight = bg_weight + focal_weight * torch.clamp(target, 0.0, 1.0) ** gamma
    if change is not None and change_weight > 0:
        weight = weight + change_weight * torch.clamp(change, 0.0, 1.0)
    reg = (elem * weight).mean()
    fa = ((pred > fa_pred_thr) & (target < fa_true_thr)).to(pred.dtype)
    fa_pen = fa * F.relu(pred - fa_pred_thr)
    return reg + fa_weight * fa_pen.mean()


def occupancy_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    occ_threshold: float = 0.3,
    pos_weight: float = 4.0,
) -> torch.Tensor:
    """Per-cell 'is this a threat?' BCE. Empty cells are the majority class."""
    y = (target >= occ_threshold).to(logits.dtype)
    pw = logits.new_tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)


def balanced_term(
    term: torch.Tensor,
    reference: torch.Tensor,
    fraction: float,
    max_boost: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale ``term`` toward ``fraction * |reference|`` without letting it dominate.

    The multiplier is detached (gradient stays in ``term``). Oversized terms are
    always scaled down to the target. Undersized terms may be boosted at most
    ``max_boost`` so a near-zero term cannot explode.
    """
    if fraction <= 0:
        return term * 0.0
    ref = reference.detach().abs()
    mag = term.detach().abs().clamp_min(eps)
    target = fraction * ref
    mult = (target / mag).clamp(max=max_boost)
    return term * mult


def direction_scores_torch(
    heatmap: torch.Tensor,
    row_weights: list[float],
    left_cols: list[int],
    center_cols: list[int],
    right_cols: list[int],
) -> dict[str, torch.Tensor]:
    """Batched LEFT/CENTER/RIGHT scores; same definition as ``warning.direction_scores``."""
    w = torch.as_tensor(row_weights, device=heatmap.device, dtype=heatmap.dtype).view(1, -1, 1)
    weighted = heatmap * w

    def group_max(cols: list[int]) -> torch.Tensor:
        if not cols:
            return heatmap.new_zeros(heatmap.shape[0])
        return weighted[:, :, cols].amax(dim=(1, 2))

    return {
        "LEFT": group_max(left_cols),
        "CENTER": group_max(center_cols),
        "RIGHT": group_max(right_cols),
    }


def readout_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    warning: dict[str, Any],
) -> torch.Tensor:
    """SmoothL1 on the three wearable direction scores."""
    pred_s = direction_scores_torch(
        pred,
        warning["row_weights"],
        warning["left_cols"],
        warning["center_cols"],
        warning["right_cols"],
    )
    true_s = direction_scores_torch(
        target,
        warning["row_weights"],
        warning["left_cols"],
        warning["center_cols"],
        warning["right_cols"],
    )
    parts = [F.smooth_l1_loss(pred_s[k], true_s[k], beta=0.1) for k in ("LEFT", "CENTER", "RIGHT")]
    return (parts[0] + parts[1] + parts[2]) / 3.0


def stop_focal_bce(
    pred: torch.Tensor,
    target: torch.Tensor,
    warning: dict[str, Any],
    temperature: float = 0.05,
    gamma: float = 2.0,
    fn_weight: float = 2.5,
) -> torch.Tensor:
    """Differentiable STOP vs not-STOP using the same max direction score as ``classify``."""
    pred_s = direction_scores_torch(
        pred,
        warning["row_weights"],
        warning["left_cols"],
        warning["center_cols"],
        warning["right_cols"],
    )
    true_s = direction_scores_torch(
        target,
        warning["row_weights"],
        warning["left_cols"],
        warning["center_cols"],
        warning["right_cols"],
    )
    s_pred = torch.stack([pred_s["LEFT"], pred_s["CENTER"], pred_s["RIGHT"]], dim=0).max(dim=0).values
    s_true = torch.stack([true_s["LEFT"], true_s["CENTER"], true_s["RIGHT"]], dim=0).max(dim=0).values
    theta = float(warning["stop_threshold"])
    p = torch.sigmoid((s_pred - theta) / max(temperature, 1e-4))
    y = (s_true >= theta).to(p.dtype)
    p = p.clamp(1e-6, 1.0 - 1e-6)
    pos = y * ((1.0 - p) ** gamma) * (-p.log())
    neg = (1.0 - y) * (p ** gamma) * (-(1.0 - p).log())
    weight = torch.where(y > 0.5, torch.full_like(y, fn_weight), torch.ones_like(y))
    return (weight * (pos + neg)).mean()


def student_loss(
    out: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    warning: dict[str, Any],
    *,
    focal_weight: float = 4.0,
    focal_gamma: float = 2.0,
    change_weight: float = 2.0,
    now_heatmap_weight: float = 0.5,
    smooth_l1_beta: float = 0.1,
    readout_frac: float = 0.40,
    stop_frac: float = 0.15,
    stop_fn_weight: float = 2.5,
    stop_temperature: float = 0.05,
    stop_focal_gamma: float = 2.0,
    latent_heatmap_frac: float = 0.30,
    occupancy_frac: float = 0.20,
    occupancy_threshold: float = 0.3,
    occupancy_pos_weight: float = 4.0,
    bg_weight: float = 0.05,
    ignore_below: float = 0.15,
    fa_weight: float = 1.0,
    fa_pred_thr: float = 0.45,
    fa_true_thr: float = 0.2,
    jepa_frac: float = 0.10,
    jepa_lambda: float = 0.0,
    balance_max_boost: float = 5.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full student objective. Auxiliary terms are balanced to fractions of L(H+)."""
    h_plus = out["h_plus_hat"]
    h_now = out["h_now_hat"]
    target_plus = batch["h_future"]
    target_now = batch["h_now"]
    change = (target_plus - target_now).abs()
    hm_kw = dict(
        focal_weight=focal_weight,
        gamma=focal_gamma,
        change_weight=change_weight,
        beta=smooth_l1_beta,
        bg_weight=bg_weight,
        ignore_below=ignore_below,
        fa_weight=fa_weight,
        fa_pred_thr=fa_pred_thr,
        fa_true_thr=fa_true_thr,
    )

    l_h_plus = collision_heatmap_loss(h_plus, target_plus, change=change, **hm_kw)
    now_kw = {k: v for k, v in hm_kw.items() if k != "change_weight"}
    l_h_now = collision_heatmap_loss(h_now, target_now, change_weight=0.0, **now_kw)

    l_read_raw = readout_loss(h_plus, target_plus, warning)
    l_stop_raw = stop_focal_bce(
        h_plus,
        target_plus,
        warning,
        temperature=stop_temperature,
        gamma=stop_focal_gamma,
        fn_weight=stop_fn_weight,
    )
    l_read = balanced_term(l_read_raw, l_h_plus, readout_frac, max_boost=balance_max_boost)
    l_stop = balanced_term(l_stop_raw, l_h_plus, stop_frac, max_boost=balance_max_boost)

    parts: dict[str, torch.Tensor] = {
        "h_plus": l_h_plus,
        "h_now": l_h_now,
        "readout_raw": l_read_raw,
        "readout": l_read,
        "stop_raw": l_stop_raw,
        "stop": l_stop,
    }

    loss = l_h_plus + now_heatmap_weight * l_h_now + l_read + l_stop

    if "h_plus_latent" in out:
        l_lat_raw = collision_heatmap_loss(
            out["h_plus_latent"], target_plus, change=change, **hm_kw
        )
        l_lat = balanced_term(l_lat_raw, l_h_plus, latent_heatmap_frac, max_boost=balance_max_boost)
        loss = loss + l_lat
        parts["latent_raw"] = l_lat_raw
        parts["latent"] = l_lat
    else:
        parts["latent_raw"] = l_h_plus.new_zeros(())
        parts["latent"] = l_h_plus.new_zeros(())

    if "occ_plus_logits" in out:
        l_occ_raw = occupancy_bce(
            out["occ_plus_logits"],
            target_plus,
            occ_threshold=occupancy_threshold,
            pos_weight=occupancy_pos_weight,
        )
        if "occ_now_logits" in out:
            l_occ_raw = l_occ_raw + occupancy_bce(
                out["occ_now_logits"],
                target_now,
                occ_threshold=occupancy_threshold,
                pos_weight=occupancy_pos_weight,
            )
        l_occ = balanced_term(l_occ_raw, l_h_plus, occupancy_frac, max_boost=balance_max_boost)
        loss = loss + l_occ
        parts["occ_raw"] = l_occ_raw
        parts["occ"] = l_occ
    else:
        parts["occ_raw"] = l_h_plus.new_zeros(())
        parts["occ"] = l_h_plus.new_zeros(())

    if jepa_lambda > 0 and "z_plus" in batch:
        l_jepa_raw = jepa_distance(out["z_plus_hat"], batch["z_plus"])
        frac = jepa_frac if jepa_frac > 0 else float(jepa_lambda)
        l_jepa = balanced_term(l_jepa_raw, l_h_plus, frac, max_boost=balance_max_boost)
        loss = loss + l_jepa
        parts["jepa_raw"] = l_jepa_raw
        parts["jepa"] = l_jepa
    else:
        parts["jepa_raw"] = l_h_plus.new_zeros(())
        parts["jepa"] = l_h_plus.new_zeros(())

    parts["total"] = loss
    return loss, parts
