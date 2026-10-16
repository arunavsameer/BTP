"""Delta / residual supervision helper."""

from __future__ import annotations

import torch

from .heatmap import target_weighted_mse


def residual_target(h_future: torch.Tensor, h_now_hat: torch.Tensor) -> torch.Tensor:
    return (h_future.float() - h_now_hat.detach().float()).clamp(-1.0, 1.0)


def delta_loss(
    delta_hat: torch.Tensor,
    h_future: torch.Tensor,
    h_now_hat: torch.Tensor,
    *,
    sample_mask: torch.Tensor | None = None,
    row_weights: torch.Tensor | None = None,
    floor: float = 0.05,
    scale: float = 4.0,
    gamma: float = 2.0,
    residual_mix: float = 0.5,
) -> torch.Tensor:
    delta_star = residual_target(h_future, h_now_hat)
    extra = residual_mix * delta_star.abs()
    return target_weighted_mse(
        delta_hat,
        delta_star,
        sample_mask=sample_mask,
        row_weights=row_weights,
        floor=floor,
        scale=scale,
        gamma=gamma,
        extra=extra,
    )
