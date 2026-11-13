"""Lightweight depthwise-separable RGB students (160×160, 3 or 5 frames)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.models.heatmap_decoder import HeatmapDecoder

MAX_TOTAL_PARAMS = 700_000


def _gn_groups(channels: int) -> int:
    c = int(channels)
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class DWBlock(nn.Module):
    def __init__(self, c: int, expand: int = 2):
        super().__init__()
        mid = c * expand
        self.pw1 = nn.Conv2d(c, mid, 1)
        self.dw = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.pw2 = nn.Conv2d(mid, c, 1)
        self.gn = nn.GroupNorm(_gn_groups(c), c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.pw1(x))
        y = self.act(self.dw(y))
        y = self.pw2(y)
        return self.act(self.gn(x + y))


def _make_stages(width: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Conv2d(3, width, 3, stride=2, padding=1), nn.GELU()]
    c = width
    for i in range(depth):
        layers.append(DWBlock(c, expand=2 if i < depth - 1 else 1))
        if i % 2 == 1:
            layers += [nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.GELU()]
            c *= 2
    return nn.Sequential(*layers)


@dataclass
class ArchConfig:
    name: str
    budget: Literal["micro", "base", "large"]
    temporal: Literal["concat", "shared"]
    width: int
    depth: int
    fine_channels: int = 128
    z_channels: int = 32
    out_grid: int = 5


class StudentRGBNet(nn.Module):
    """Spatial map preserved until 5×5 pooling / decoder."""

    def __init__(self, cfg: ArchConfig, n_frames: int, input_size: int = 160):
        super().__init__()
        self.cfg = cfg
        self.n_frames = int(n_frames)
        self.input_size = int(input_size)
        self.fine_channels = cfg.fine_channels
        self.z_channels = cfg.z_channels
        self.feature_grid = cfg.out_grid
        if cfg.temporal == "concat":
            stem_in = 3 * self.n_frames
            self.encoder = _make_stages(cfg.width, cfg.depth)
            self.stem = nn.Conv2d(stem_in, 3, 1)
            self.head = nn.Sequential(
                nn.Conv2d(self._encoder_out_ch(cfg), cfg.fine_channels, 1),
                nn.GELU(),
            )
            self.frame_encoder = None
        else:
            self.stem = None
            self.frame_encoder = _make_stages(cfg.width, cfg.depth)
            out_ch = self._encoder_out_ch(cfg)
            self.temporal_fuse = nn.Sequential(
                nn.Conv2d(out_ch * self.n_frames, out_ch, 1),
                nn.GELU(),
                nn.Conv2d(out_ch, cfg.fine_channels, 1),
                nn.GELU(),
            )
            self.encoder = None
            self.head = None
        self.z_proj = nn.Sequential(
            nn.Conv2d(cfg.fine_channels, cfg.fine_channels, 1),
            nn.GELU(),
            nn.Conv2d(cfg.fine_channels, cfg.z_channels, 1),
        )
        self.decoder = HeatmapDecoder(
            z_channels=cfg.fine_channels,
            dilated=True,
            out_grid=cfg.out_grid,
            pool="max",
        )
        self.close_head = nn.Conv2d(cfg.z_channels, 1, kernel_size=1)

    def _encoder_out_ch(self, cfg: ArchConfig) -> int:
        c = cfg.width
        for i in range(cfg.depth):
            if i % 2 == 1:
                c *= 2
        return c

    def _spatial_feats(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, 3, H, W] in [0,1] → fine [B, C, h, w]."""
        b, t, c, h, w = x.shape
        if self.cfg.temporal == "concat":
            flat = x.reshape(b, t * c, h, w)
            assert self.stem is not None and self.encoder is not None
            y = self.encoder(self.stem(flat))
            assert self.head is not None
            return self.head(y)
        assert self.frame_encoder is not None
        feats = []
        for i in range(t):
            feats.append(self.frame_encoder(x[:, i]))
        cat = torch.cat(feats, dim=1)
        return self.temporal_fuse(cat)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        fine = self._spatial_feats(clip)
        g = self.feature_grid
        if fine.shape[-1] != g or fine.shape[-2] != g:
            fine_grid = F.adaptive_avg_pool2d(fine, g)
        else:
            fine_grid = fine
        z = self.z_proj(fine)
        if z.shape[-1] != g:
            z = F.adaptive_avg_pool2d(z, g)
        h, logits = self.decoder.forward_with_logits(fine)
        close = self.close_head(z).squeeze(1)
        return {
            "F": fine,
            "Z_t": z,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
            "Hhat_close_logits": close,
        }

    def parameter_audit(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        enc_mod = self.frame_encoder or self.encoder
        enc = sum(p.numel() for p in enc_mod.parameters()) if enc_mod else 0
        return {"total": total, "encoder": enc, "decoder": dec, "config": self.cfg.__dict__}


def preset_architectures() -> list[ArchConfig]:
    """Micro/Base unchanged from first sweep; Large redesigned to ≤700K total."""
    return [
        ArchConfig("micro_concat", "micro", "concat", width=32, depth=4),
        ArchConfig("micro_shared", "micro", "shared", width=32, depth=4),
        ArchConfig("base_concat", "base", "concat", width=48, depth=5),
        ArchConfig("base_shared", "base", "shared", width=48, depth=5),
        ArchConfig("large_concat", "large", "concat", width=56, depth=5),
        ArchConfig("large_shared", "large", "shared", width=54, depth=4),
    ]


def assert_param_budget(model: nn.Module, budget: int = MAX_TOTAL_PARAMS) -> int:
    n = sum(p.numel() for p in model.parameters())
    if n > budget:
        raise ValueError(f"student has {n} parameters > budget {budget}")
    return n
