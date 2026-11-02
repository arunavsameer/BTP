"""Consistency mixer: trust current raw maps when they agree, else hold the last final.

``alpha = exp(-std / tau)`` with a learnable ``tau``. ``std`` is occupancy-weighted
per-cell spread over a short window of *raw* 5x5 maps, never ``|raw - final|``.
The first snapshot of a clip copies raw. ``tau`` is initialized high so training
starts near ``alpha ≈ 1``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def occupancy_weighted_std(maps: torch.Tensor) -> torch.Tensor:
    """Scalar std of a raw window ``[W, 5, 5]``. Empty cells do not dominate.

    Per-cell std over the window, weighted by that cell's mean occupancy.
    A single map has std 0.
    """
    if maps.ndim != 3:
        raise ValueError(f"expected [W, H, W] raw maps, got {tuple(maps.shape)}")
    if maps.shape[0] < 2:
        return maps.new_zeros(())
    mean = maps.mean(dim=0)
    std_cell = maps.std(dim=0, unbiased=False)
    weight = mean.clamp(min=0.0)
    wsum = weight.sum()
    unweighted = std_cell.mean()
    weighted = (std_cell * weight).sum() / wsum.clamp(min=1e-6)
    return torch.where(wsum > 1e-6, weighted, unweighted)


def safe_logit(prob: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    p = prob.clamp(eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


class ConsistencySmoother(nn.Module):
    """Causal mix of raw heatmaps. State lives on the module and is clip-local."""

    def __init__(self, window: int = 4, tau_init: float = 0.5):
        super().__init__()
        self.window = int(window)
        if self.window < 2:
            raise ValueError(f"smoother.window must be >= 2, got {self.window}")
        tau_init = float(tau_init)
        if tau_init <= 0.0:
            raise ValueError(f"smoother.tau_init must be > 0, got {tau_init}")
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init), dtype=torch.float32))
        self.reset()

    def reset(self) -> None:
        self._raw_hist: list[torch.Tensor] = []
        self._final: torch.Tensor | None = None
        self._clip_id: str | None = None

    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(min=1e-4)

    def forward(
        self,
        raw_probs: torch.Tensor,
        raw_logits: torch.Tensor | None = None,
        clip_ids: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        del raw_logits
        if raw_probs.ndim != 3:
            raise ValueError(f"raw_probs expected [B, 5, 5], got {tuple(raw_probs.shape)}")
        batch = int(raw_probs.shape[0])
        if clip_ids is None:
            self.reset()
            clip_ids = ["__anon__"] * batch
        elif len(clip_ids) != batch:
            raise ValueError(f"clip_ids length {len(clip_ids)} != batch {batch}")

        hist = list(self._raw_hist)
        final = self._final
        clip_id = self._clip_id
        mixed: list[torch.Tensor] = []
        alphas: list[torch.Tensor] = []
        tau = self.tau().to(dtype=raw_probs.dtype)

        for i in range(batch):
            cid = str(clip_ids[i])
            if cid != clip_id:
                hist = []
                final = None
                clip_id = cid
            raw_i = raw_probs[i]
            window = (hist + [raw_i])[-self.window :]
            std = occupancy_weighted_std(torch.stack(window, dim=0))
            alpha = torch.exp(-std / tau)
            if final is None:
                final_i = raw_i
                alpha_i = raw_i.new_ones(())
            else:
                alpha_i = alpha
                final_i = alpha_i * raw_i + (1.0 - alpha_i) * final
            mixed.append(final_i)
            alphas.append(alpha_i.reshape(()))
            hist = window
            final = final_i

        self._raw_hist = [t.detach() for t in hist]
        self._final = None if final is None else final.detach()
        self._clip_id = clip_id

        probs = torch.stack(mixed, dim=0)
        alpha = torch.stack(alphas, dim=0)
        return {
            "Hhat_1s": probs,
            "Hhat_1s_logits": safe_logit(probs),
            "alpha": alpha,
        }
