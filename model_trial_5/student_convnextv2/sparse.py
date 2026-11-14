"""Submanifold sparse 3D convs (ConvNeXt-V2 / FCMAE).

Masking a dense conv before and after is numerically the same as
MinkowskiEngine submanifold sparse convolution, without that dependency.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def random_patch_mask(
    batch: int,
    grid: int = 16,
    *,
    mask_ratio: float = 0.6,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Visible=1, masked=0 on a square grid. Shape [B,1,1,G,G] (broadcast over T)."""
    n = int(grid) * int(grid)
    n_mask = int(round(n * float(mask_ratio)))
    n_mask = min(max(n_mask, 0), n)
    noise = torch.rand(int(batch), n, device=device)
    ids = noise.argsort(dim=1)
    vis = torch.ones(int(batch), n, device=device, dtype=dtype)
    if n_mask > 0:
        vis.scatter_(1, ids[:, :n_mask], 0)
    return vis.view(int(batch), 1, 1, int(grid), int(grid))


def resize_mask(mask: torch.Tensor, size_thw: tuple[int, int, int]) -> torch.Tensor:
    """Nearest-resize a [B,1,T,H,W] (or broadcastable) mask to ``size_thw``."""
    t, h, w = (int(x) for x in size_thw)
    if mask.ndim != 5:
        raise ValueError(f"mask must be 5D, got {tuple(mask.shape)}")
    if mask.shape[2] == t and mask.shape[3] == h and mask.shape[4] == w:
        return mask
    if mask.shape[2] == 1 and t != 1:
        mask = mask.expand(-1, -1, t, -1, -1).contiguous()
        if mask.shape[3] == h and mask.shape[4] == w:
            return mask
    return F.interpolate(mask.float(), size=(t, h, w), mode="nearest").to(dtype=mask.dtype)


def apply_submanifold(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x
    m = resize_mask(mask, x.shape[2:])
    return x * m
