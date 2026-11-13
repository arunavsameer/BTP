"""V2 dataset extras: hazard sampling weights and geometry-consistent augs."""

from __future__ import annotations

import torch
from torch.utils.data import WeightedRandomSampler


def hazard_sample_weights(peaks: torch.Tensor | list[float]) -> torch.Tensor:
    p = peaks if torch.is_tensor(peaks) else torch.tensor(peaks, dtype=torch.float32)
    p = p.float().view(-1)
    w = torch.ones_like(p)
    w = torch.where(p >= 0.25, torch.full_like(w, 1.5), w)
    w = torch.where(p >= 0.60, torch.full_like(w, 3.0), w)
    return w


def weighted_sampler(weights: torch.Tensor) -> WeightedRandomSampler:
    return WeightedRandomSampler(weights.double(), num_samples=int(weights.numel()), replacement=True)


def hflip_batch(batch: dict) -> dict:
    batch["clip"] = batch["clip"].flip(-1)
    for k in ("F_teacher", "Z_teacher_16", "P_teacher_5", "H_teacher", "gt_1s", "h_1s", "h_now"):
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].flip(-1)
    return batch


def color_jitter_clip(clip: torch.Tensor, brightness: float = 0.15, contrast: float = 0.15, noise_std: float = 0.02) -> torch.Tensor:
    """clip: [B, T, C, H, W] in 0..1. Teacher tensors are not changed."""
    b_factor = 1.0 + (2.0 * torch.rand((), device=clip.device, dtype=clip.dtype) - 1.0) * float(brightness)
    c_factor = 1.0 + (2.0 * torch.rand((), device=clip.device, dtype=clip.dtype) - 1.0) * float(contrast)
    mean = clip.mean(dim=(-2, -1), keepdim=True)
    x = (clip - mean) * c_factor + mean
    x = x * b_factor
    if float(noise_std) > 0:
        x = x + torch.randn_like(x) * float(noise_std)
    return x.clamp(0.0, 1.0)


def augment_batch(batch: dict, *, flip_p: float = 0.5) -> dict:
    if torch.rand(()) < float(flip_p):
        batch = hflip_batch(batch)
    batch["clip"] = color_jitter_clip(batch["clip"])
    return batch
