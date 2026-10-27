"""Beta threat distributions: proper scoring + evidential precision + noisy-OR STOP.

Mean-precision Beta regression (Ferrari & Cribari-Neto 2004). Each cell's
threat H ~ Beta(α, β) with μ = α/(α+β) = predicted heatmap and precision
φ = α+β = 1 + softplus(raw). NLL is a strictly proper scoring rule
(Gneiting & Raftery 2007). The evidential penalty |μ−y|φ shrinks precision
when the mean is wrong (Amini et al., NeurIPS 2020 analogue). Frame STOP
probability is a noisy-OR of per-cell P(H ≥ 0.60) using a normal
approximation of the Beta (Kendall & Gal, NeurIPS 2017 style).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


EPS = 1e-4
MIN_CONC = 1e-3
SQRT2 = math.sqrt(2.0)


def precision_from_logits(raw: torch.Tensor) -> torch.Tensor:
    """φ = 1 + softplus(raw) ∈ [1, 64]. Cap keeps lgamma stable."""
    return (1.0 + F.softplus(raw.float())).clamp(max=64.0)


def concentrations(mu: torch.Tensor, precision: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mu = mu.float().clamp(EPS, 1.0 - EPS)
    phi = precision.float().clamp(min=1.0)
    alpha = (mu * phi).clamp(min=MIN_CONC)
    beta = ((1.0 - mu) * phi).clamp(min=MIN_CONC)
    return alpha, beta


def beta_nll(
    mu: torch.Tensor,
    precision_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    cell_weight: torch.Tensor | None = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Mean NLL of target under Beta(μφ, (1−μ)φ). Differentiable via lgamma."""
    phi = precision_from_logits(precision_logits)
    alpha, beta = concentrations(mu, phi)
    y = target.float().clamp(eps, 1.0 - eps)
    # log B(α,β) = lgamma(α)+lgamma(β)-lgamma(α+β)
    log_norm = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    nll = -( (alpha - 1.0) * y.log() + (beta - 1.0) * (1.0 - y).log() - log_norm )
    w = torch.ones_like(nll)
    if cell_weight is not None:
        w = w * cell_weight.to(dtype=w.dtype, device=w.device)
    if sample_mask is not None:
        sm = sample_mask.to(dtype=w.dtype, device=w.device).view(-1, 1, 1)
        w = w * sm
        if float(sample_mask.sum()) <= 0:
            return nll.new_zeros(())
    denom = w.sum().clamp(min=eps)
    return (nll * w).sum() / denom


def evidential_mistake_penalty(
    mu: torch.Tensor,
    precision_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pay for being confident and wrong: mean |μ − y| φ."""
    phi = precision_from_logits(precision_logits)
    err = (mu.float() - target.float()).abs() * phi
    if sample_mask is not None:
        sm = sample_mask.to(dtype=err.dtype, device=err.device).view(-1, 1, 1)
        err = err * sm
        if float(sample_mask.sum()) <= 0:
            return err.new_zeros(())
        return err.sum() / sm.sum().clamp(min=1.0) / float(err.shape[-1] * err.shape[-2])
    return err.mean()


def cell_stop_probability(
    mu: torch.Tensor,
    precision_logits: torch.Tensor,
    *,
    threshold: float = 0.60,
) -> torch.Tensor:
    """P(H ≥ t) via Normal(μ, Var[Beta]) CDF. Fully differentiable."""
    phi = precision_from_logits(precision_logits)
    mu = mu.float().clamp(EPS, 1.0 - EPS)
    var = mu * (1.0 - mu) / (phi + 1.0)
    sigma = var.clamp(min=1e-6).sqrt()
    z = (float(threshold) - mu) / sigma
    # 1 - Φ(z) = 0.5 * erfc(z / sqrt(2))
    return 0.5 * torch.erfc(z / SQRT2)


def noisy_or_stop(
    mu: torch.Tensor,
    precision_logits: torch.Tensor,
    *,
    threshold: float = 0.60,
    sample_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Frame-level P(any cell ≥ t) under independent Betas. [B]."""
    p = cell_stop_probability(mu, precision_logits, threshold=threshold).clamp(1e-6, 1.0 - 1e-6)
    if sample_mask is not None:
        sm = sample_mask.to(dtype=p.dtype, device=p.device).view(-1, 1, 1)
        p = p * sm
    log_surv = torch.log1p(-p).flatten(1).sum(dim=1)
    return (1.0 - log_surv.exp()).clamp(0.0, 1.0)


def prob_stop_hinge(
    mu: torch.Tensor,
    precision_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    gt_stop: float = 0.60,
    target_prob: float = 0.75,
) -> torch.Tensor:
    """If GT peak is STOP, push noisy-OR P(any ≥ 0.60) up to target_prob."""
    gt_peak = target.float().flatten(1).max(dim=1).values
    need = gt_peak >= float(gt_stop)
    if sample_mask is not None:
        need = need & sample_mask.to(dtype=torch.bool, device=mu.device).view(-1)
    if not bool(need.any()):
        return mu.new_zeros(())
    p = noisy_or_stop(mu, precision_logits, threshold=gt_stop, sample_mask=sample_mask)
    return (float(target_prob) - p[need]).clamp(min=0.0).mean()


def prob_safe_hinge(
    mu: torch.Tensor,
    precision_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    caution: float = 0.25,
    cap_prob: float = 0.15,
    stop_threshold: float = 0.60,
) -> torch.Tensor:
    """If GT peak is SAFE, push noisy-OR P(any ≥ 0.60) below cap_prob."""
    gt_peak = target.float().flatten(1).max(dim=1).values
    need = gt_peak < float(caution)
    if sample_mask is not None:
        need = need & sample_mask.to(dtype=torch.bool, device=mu.device).view(-1)
    if not bool(need.any()):
        return mu.new_zeros(())
    p = noisy_or_stop(mu, precision_logits, threshold=stop_threshold, sample_mask=sample_mask)
    return (p[need] - float(cap_prob)).clamp(min=0.0).mean()
