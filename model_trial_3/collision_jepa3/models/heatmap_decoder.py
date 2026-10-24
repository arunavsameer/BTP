"""Heatmap decoder. Fine-grid decode then pool to 5×5; residual Δ stays on Z_t."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _spatial_heatmap_tower(z_channels: int, hidden: int, dilated: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(z_channels, hidden, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    ]
    if dilated:
        # Dilation 2 on the decode grid (16×16 or 5×5). Independent per-cell
        # sigmoid (not spatial softmax) so the blob is not 1-hotted.
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        ]
    layers.append(nn.Conv2d(hidden, 1, kernel_size=1))
    return nn.Sequential(*layers)


def adaptive_bins(src: int, dst: int) -> list[tuple[int, int]]:
    """Same windows as adaptive max/avg pool: [floor(i*src/dst), ceil((i+1)*src/dst))."""
    bins: list[tuple[int, int]] = []
    for i in range(int(dst)):
        start = (i * src) // dst
        end = ((i + 1) * src + dst - 1) // dst
        if end <= start:
            end = min(start + 1, src)
        bins.append((start, end))
    return bins


def adaptive_attn_pool(values: torch.Tensor, logits: torch.Tensor, grid: int) -> torch.Tensor:
    """Softmax-weighted pool of `values` inside each coarse cell. [B,H,W] → [B,G,G]."""
    v = values.unsqueeze(1) if values.dim() == 3 else values
    lg = logits.unsqueeze(1) if logits.dim() == 3 else logits
    _, _, hf, wf = v.shape
    g = int(grid)
    out = v.new_zeros(v.shape[0], 1, g, g)
    for i, (rs, re) in enumerate(adaptive_bins(hf, g)):
        for j, (cs, ce) in enumerate(adaptive_bins(wf, g)):
            patch_l = lg[:, :, rs:re, cs:ce].flatten(2)
            patch_v = v[:, :, rs:re, cs:ce].flatten(2)
            w = torch.softmax(patch_l, dim=2)
            out[:, :, i, j] = (patch_v * w).sum(dim=2)
    return out.squeeze(1)


def pool_heatmap(h_fine: torch.Tensor, grid: int, pool: str, logits: torch.Tensor | None = None) -> torch.Tensor:
    """Reduce a fine heatmap [B,H,W] to [B,grid,grid]. Max = “any object in this cell”."""
    if h_fine.shape[-2] == grid and h_fine.shape[-1] == grid:
        return h_fine
    mode = str(pool)
    x = h_fine.unsqueeze(1)
    if mode == "max":
        return F.adaptive_max_pool2d(x, grid).squeeze(1)
    if mode == "avg":
        return F.adaptive_avg_pool2d(x, grid).squeeze(1)
    if mode == "attn":
        if logits is None:
            raise ValueError("attn heatmap pool needs logits")
        return adaptive_attn_pool(h_fine, logits, grid)
    raise ValueError(f"unsupported heatmap pool {mode}")


class HeatmapDecoder(nn.Module):
    def __init__(
        self,
        z_channels: int = 32,
        hidden: int = 48,
        dilated: bool = True,
        out_grid: int = 5,
        pool: str = "max",
    ):
        super().__init__()
        mode = str(pool)
        if mode not in {"max", "avg", "attn"}:
            raise ValueError(f"unsupported heatmap pool {mode}")
        self.dilated = bool(dilated)
        self.out_grid = int(out_grid)
        self.pool = mode
        self.net = _spatial_heatmap_tower(z_channels, hidden, self.dilated)
        self.pool_score = nn.Conv2d(z_channels, 1, kernel_size=1) if mode == "attn" else None

    def forward_with_logits(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Heatmap in [0, 1] plus 5×5 logits for BCE-with-logits.

        For max-pool, sigmoid(max(logits)) = max(sigmoid(logits)), so pooling
        logits then sigmoid matches the old pool-after-sigmoid maps.
        """
        logits_fine = self.net(z)
        if self.pool == "max":
            if logits_fine.shape[-2] != self.out_grid or logits_fine.shape[-1] != self.out_grid:
                pooled = F.adaptive_max_pool2d(logits_fine, self.out_grid)
            else:
                pooled = logits_fine
            h = torch.sigmoid(pooled).squeeze(1)
            return h, pooled.squeeze(1)
        h_fine = torch.sigmoid(logits_fine).squeeze(1)
        attn_logits = self.pool_score(z) if self.pool_score is not None else None
        h = pool_heatmap(h_fine, self.out_grid, self.pool, logits=attn_logits)
        return h, logits_fine.squeeze(1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, _logits = self.forward_with_logits(z)
        return h


class HeatmapDelta(nn.Module):
    def __init__(self, z_channels: int = 32, hidden: int = 48, dilated: bool = True):
        super().__init__()
        self.dilated = bool(dilated)
        self.net = _spatial_heatmap_tower(z_channels, hidden, self.dilated)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(z)).squeeze(1)


def copy_residual_future(h_now: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Hhat_h = clamp(sg(Hhat_now) + Delta_h, 0, 1). Shared by all horizons."""
    return (h_now.detach() + delta).clamp(0.0, 1.0)
