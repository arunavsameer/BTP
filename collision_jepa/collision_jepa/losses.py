"""Loss functions.

The heatmap loss fights the center-corridor prior: a plain MSE model can score well
by predicting "the corridor is mildly dangerous" while ignoring real threats. We add
a focal-style high-threat weighting so cells with high true threat dominate.
"""

from __future__ import annotations

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
