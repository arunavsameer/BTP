"""Shared-weight student with masked multi-pattern temporal fusion (max 5 frames)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.models.heatmap_decoder import HeatmapDecoder

from .cnn import DWBlock, MAX_TOTAL_PARAMS, _gn_groups, _make_stages, assert_param_budget
from .frames import FRAME_CANDIDATES

MAX_FRAMES = 5
ALL_OFFSETS = sorted({o for pid in ("F1", "F2", "F3", "F4", "F5") for o in FRAME_CANDIDATES[pid]})


class TemporalSupernet(nn.Module):
    def __init__(self, *, width: int = 32, depth: int = 4, fine_channels: int = 128, z_channels: int = 32):
        super().__init__()
        self.max_frames = MAX_FRAMES
        self.fine_channels = fine_channels
        self.z_channels = z_channels
        self.feature_grid = 5
        self.frame_encoder = _make_stages(width, depth)
        enc_out = self._encoder_out_ch(width, depth)
        self.offset_embed = nn.Sequential(nn.Linear(1, enc_out), nn.GELU(), nn.Linear(enc_out, enc_out))
        self.fuse = nn.Sequential(
            nn.Conv2d(enc_out, fine_channels, 1),
            nn.GELU(),
        )
        self.z_proj = nn.Sequential(
            nn.Conv2d(fine_channels, fine_channels, 1),
            nn.GELU(),
            nn.Conv2d(fine_channels, z_channels, 1),
        )
        self.decoder = HeatmapDecoder(
            z_channels=fine_channels,
            dilated=True,
            out_grid=5,
            pool="max",
        )
        self.close_head = nn.Conv2d(z_channels, 1, kernel_size=1)

    @staticmethod
    def _encoder_out_ch(width: int, depth: int) -> int:
        c = width
        for i in range(depth):
            if i % 2 == 1:
                c *= 2
        return c

    def forward(
        self,
        clip: torch.Tensor,
        *,
        mask: torch.Tensor,
        offsets_norm: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """clip [B,T,3,H,W], mask [B,T] float/bool, offsets_norm [B,T] in [-1,1] (invalid masked)."""
        b, t, _, h, w = clip.shape
        assert t == self.max_frames
        m = mask.float()
        pooled = 0.0
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0).view(b, 1, 1, 1)
        for i in range(t):
            fi = self.frame_encoder(clip[:, i])
            emb = self.offset_embed(offsets_norm[:, i : i + 1]).unsqueeze(-1).unsqueeze(-1)
            fi = fi + emb
            pooled = pooled + fi * m[:, i].view(b, 1, 1, 1)
        fine = self.fuse(pooled / denom)
        z = self.z_proj(fine)
        if fine.shape[-1] != self.feature_grid:
            fine = F.adaptive_avg_pool2d(fine, self.feature_grid)
        if z.shape[-1] != self.feature_grid:
            z = F.adaptive_avg_pool2d(z, self.feature_grid)
        hmap, logits = self.decoder.forward_with_logits(fine)
        close = self.close_head(z).squeeze(1)
        return {
            "F": fine,
            "Z_t": z,
            "Hhat_1s": hmap,
            "Hhat_1s_logits": logits,
            "Hhat_close_logits": close,
        }

    def forward_pattern(self, clip: torch.Tensor, pattern: str) -> dict[str, torch.Tensor]:
        offs = FRAME_CANDIDATES[pattern]
        mask, offsets_norm = pattern_mask_and_offsets(offs, device=clip.device, batch=clip.shape[0])
        return self.forward(clip, mask=mask, offsets_norm=offsets_norm)

    def parameter_audit(self) -> dict[str, Any]:
        enc = sum(p.numel() for p in self.frame_encoder.parameters())
        fuse = sum(p.numel() for p in self.fuse.parameters()) + sum(p.numel() for p in self.offset_embed.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        other = sum(p.numel() for p in self.parameters()) - enc - fuse - dec
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "frame_encoder": enc,
            "temporal_fusion": fuse,
            "decoder_head": dec + other,
            "modules": {
                "frame_encoder": enc,
                "offset_embed": sum(p.numel() for p in self.offset_embed.parameters()),
                "fuse": sum(p.numel() for p in self.fuse.parameters()),
                "z_proj": sum(p.numel() for p in self.z_proj.parameters()),
                "decoder": dec,
                "close_head": sum(p.numel() for p in self.close_head.parameters()),
            },
        }


def pattern_mask_and_offsets(offsets: list[int], *, device: torch.device, batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = torch.zeros(batch, MAX_FRAMES, device=device)
    norm = torch.zeros(batch, MAX_FRAMES, device=device)
    span = max(abs(min(offsets)), abs(max(offsets)), 1)
    for i, o in enumerate(offsets[-MAX_FRAMES:]):
        if i >= MAX_FRAMES:
            break
        mask[:, i] = 1.0
        norm[:, i] = float(o) / float(span)
    return mask, norm


def pack_pattern_clip(flat_frames: torch.Tensor, pattern: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """flat_frames [B, len(ALL_OFFSETS), 3, H, W] in canonical ALL_OFFSETS order."""
    idx = {o: i for i, o in enumerate(ALL_OFFSETS)}
    offs = FRAME_CANDIDATES[pattern]
    b, _, c, h, w = flat_frames.shape
    out = torch.zeros(b, MAX_FRAMES, c, h, w, device=flat_frames.device, dtype=flat_frames.dtype)
    for i, o in enumerate(offs):
        out[:, i] = flat_frames[:, idx[o]]
    mask, offsets_norm = pattern_mask_and_offsets(offs, device=flat_frames.device, batch=b)
    return out, mask, offsets_norm


def estimate_macs(model: TemporalSupernet, n_valid: int, input_size: int = 160) -> int:
    """Rough MAC count via one forward (for reporting)."""
    dev = next(model.parameters()).device
    x = torch.zeros(1, MAX_FRAMES, 3, input_size, input_size, device=dev)
    mask, off = pattern_mask_and_offsets(FRAME_CANDIDATES["F1" if n_valid == 3 else "F4"], device=dev, batch=1)
    if n_valid == 3:
        mask[:, 3:] = 0
    with torch.no_grad():
        model.forward(x, mask=mask, offsets_norm=off)
    return 0  # placeholder; script fills profiled estimate


def build_supernet() -> TemporalSupernet:
    return TemporalSupernet(width=32, depth=4, fine_channels=128, z_channels=32)
