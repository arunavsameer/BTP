"""S0/S1 student models: Base RGB backbone + optional parent decoder transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.models.heatmap_decoder import HeatmapDecoder

from .cnn import ArchConfig, StudentRGBNet, preset_architectures


FDEC_FRAME_OFFSETS = [-90, -66, -42, -12, 0]


@dataclass
class FDecNormStats:
    mu: torch.Tensor  # [128, 1, 1]
    sigma: torch.Tensor  # [128, 1, 1]

    def to(self, device: str) -> FDecNormStats:
        return FDecNormStats(self.mu.to(device), self.sigma.to(device))


class FDecSpatialProjection(nn.Module):
    """20×20×128 → 16×16×128 via resize + depthwise refine."""

    def __init__(self, channels: int = 128, out_grid: int = 16):
        super().__init__()
        self.channels = int(channels)
        self.out_grid = int(out_grid)
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.out_grid or x.shape[-1] != self.out_grid:
            x = F.interpolate(x, size=(self.out_grid, self.out_grid), mode="bilinear", align_corners=False)
        x = self.act(self.dw(x))
        return self.act(self.pw(x))


def base_concat_config() -> ArchConfig:
    return next(a for a in preset_architectures() if a.name == "base_concat")


def backbone_state_keys() -> tuple[str, ...]:
    return ("stem", "encoder", "head")


def extract_backbone_state(model: StudentRGBNet) -> dict[str, torch.Tensor]:
    full = model.state_dict()
    return {k: v.clone() for k, v in full.items() if k.split(".")[0] in backbone_state_keys()}


def load_backbone_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    cur = model.state_dict()
    for k, v in state.items():
        if k in cur:
            cur[k] = v
    model.load_state_dict(cur, strict=False)


class StudentS0(nn.Module):
    """Standard student: owned heatmap decoder."""

    def __init__(self, arch: ArchConfig | None = None, n_frames: int = 5):
        super().__init__()
        arch = arch or base_concat_config()
        self.net = StudentRGBNet(arch, n_frames=n_frames, input_size=160)

    @property
    def fine_channels(self) -> int:
        return self.net.fine_channels

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.net(clip)

    def param_audit(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.net.parameters())
        dec = sum(p.numel() for p in self.net.decoder.parameters())
        return {
            "variant": "S0",
            "total_params": total,
            "backbone_params": total - dec,
            "decoder_params": dec,
            "projection_params": 0,
        }


class StudentS1(nn.Module):
    """Parent-decoder student with F_dec projection."""

    def __init__(
        self,
        arch: ArchConfig | None = None,
        n_frames: int = 5,
        *,
        parent_decoder: HeatmapDecoder,
        proj_out_grid: int = 16,
    ):
        super().__init__()
        arch = arch or base_concat_config()
        self.backbone = StudentRGBNet(arch, n_frames=n_frames, input_size=160)
        self.proj = FDecSpatialProjection(channels=arch.fine_channels, out_grid=proj_out_grid)
        self.decoder = parent_decoder
        self.fine_channels = arch.fine_channels

    def spatial_features(self, clip: torch.Tensor) -> torch.Tensor:
        return self.backbone._spatial_feats(clip)

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        fine = self.spatial_features(clip)
        f_s = self.proj(fine)
        h, logits = self.decoder.forward_with_logits(f_s)
        return {
            "F_student": f_s,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
            "F_backbone": fine,
        }

    def set_decoder_trainable(self, trainable: bool) -> None:
        for p in self.decoder.parameters():
            p.requires_grad = bool(trainable)

    def param_audit(self) -> dict[str, Any]:
        enc = sum(p.numel() for p in self.backbone.stem.parameters())
        enc += sum(p.numel() for p in self.backbone.encoder.parameters())
        enc += sum(p.numel() for p in self.backbone.head.parameters())  # type: ignore[union-attr]
        proj = sum(p.numel() for p in self.proj.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        total = enc + proj + dec
        return {
            "variant": "S1",
            "total_params": total,
            "backbone_params": enc,
            "projection_params": proj,
            "decoder_params": dec,
        }


class StudentS2(StudentS1):
    """S1 architecture; parent decoder stays frozen for all epochs (see run_fdec_pilot)."""

    def param_audit(self) -> dict[str, Any]:
        a = super().param_audit()
        a["variant"] = "S2"
        return a


def decoder_state_hash(dec: torch.nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for k, v in sorted(dec.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def build_parent_decoder(cfg, device: str = "cpu") -> HeatmapDecoder:
    from .parent import load_pair_cf_parent, resolve_parent_ckpt

    parent, _ = load_pair_cf_parent(cfg, resolve_parent_ckpt(cfg, None), device=device)
    dec = HeatmapDecoder(
        z_channels=int(cfg.get("adapter.projection_channels", 128)),
        dilated=bool(cfg.get("adapter.decoder_dilated", True)),
        out_grid=int(cfg.get("adapter.feature_grid", 5)),
        pool=str(cfg.get("adapter.heatmap_pool", "max")),
    )
    dec.load_state_dict(parent.decoder.state_dict())
    return dec


def init_shared_backbone(seed: int = 1337, n_frames: int = 5) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(seed))
    arch = base_concat_config()
    net = StudentRGBNet(arch, n_frames=n_frames)
    return extract_backbone_state(net)
