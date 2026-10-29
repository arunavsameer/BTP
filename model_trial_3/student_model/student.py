"""RGB student that predicts G's frozen F latent (JEPA predictor). Decoder is eval-only."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.models.heatmap_decoder import HeatmapDecoder
from collision_jepa3.models.vjepa_backbone import IMAGENET_MEAN, IMAGENET_STD

from .encoder import StudentEncoder

MAX_PARAMS = 1_000_000


def _teacher_bottleneck(z_channels: int = 32, proj: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(proj, proj, kernel_size=1),
        nn.GELU(),
        nn.Conv2d(proj, z_channels, kernel_size=1),
    )


def _sincos_2d(grid: int, channels: int) -> torch.Tensor:
    """Fixed 2D sine-cosine positional grid [1, C, S, S] (I-JEPA / ViT style)."""
    assert channels % 4 == 0
    s = int(grid)
    c = int(channels)
    y, x = torch.meshgrid(torch.arange(s).float(), torch.arange(s).float(), indexing="ij")
    y = y / max(s - 1, 1)
    x = x / max(s - 1, 1)
    omega = torch.exp(torch.arange(0, c // 4).float() * (-torch.log(torch.tensor(10000.0)) / (c // 4)))
    out = torch.cat(
        [
            torch.sin(y.unsqueeze(-1) * omega),
            torch.cos(y.unsqueeze(-1) * omega),
            torch.sin(x.unsqueeze(-1) * omega),
            torch.cos(x.unsqueeze(-1) * omega),
        ],
        dim=-1,
    )
    return out.permute(2, 0, 1).unsqueeze(0)


class DWSepRes(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        self.dw = nn.Conv2d(c, c, kernel_size=3, padding=1, groups=c)
        self.pw = nn.Conv2d(c, c, kernel_size=1)
        self.gn = nn.GroupNorm(8, c)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.gn(self.pw(self.dw(x))))


class JEPAPredictor(nn.Module):
    """Narrow residual predictor: encoder F → teacher F. Identity at init (zero last 1×1)."""

    def __init__(self, channels: int = 128, grid: int = 16):
        super().__init__()
        self.block = DWSepRes(channels)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.register_buffer("pos", _sincos_2d(grid, channels), persistent=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = x + self.pos.to(device=x.device, dtype=x.dtype)
        if mask is not None:
            h = h * mask
        return x + self.proj(self.block(h))


class CollisionStudent(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        proj = 128
        z_ch = 32
        grid = 5
        if cfg is not None:
            proj = int(cfg.get("adapter.projection_channels", 128))
            z_ch = int(cfg.get("adapter.z_channels", 32))
            grid = int(cfg.get("adapter.feature_grid", 5))
        self.z_channels = z_ch
        self.feature_grid = grid
        self.fine_channels = proj
        self.encoder = StudentEncoder(out_channels=proj)
        self.predictor = JEPAPredictor(proj, grid=16)
        self.bottleneck = _teacher_bottleneck(z_ch, proj)
        self.norm = nn.LayerNorm(z_ch)
        self.decoder = HeatmapDecoder(
            z_channels=proj,
            dilated=True if cfg is None else bool(cfg.get("adapter.decoder_dilated", True)),
            out_grid=grid,
            pool="max" if cfg is None else str(cfg.get("adapter.heatmap_pool", "max")),
        )
        self.close_head = nn.Conv2d(z_ch, 1, kernel_size=1)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)
        self._freeze_teacher_heads()

    def _freeze_teacher_heads(self) -> None:
        for mod in (self.bottleneck, self.norm, self.decoder, self.close_head):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_teacher_heads()
        return self

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(device=clip.device, dtype=clip.dtype)) / self._std.to(
            device=clip.device, dtype=clip.dtype
        )

    def encode_context(self, clip: torch.Tensor) -> torch.Tensor:
        x = self.normalize(clip).permute(0, 2, 1, 3, 4).contiguous()
        return self.encoder(x)

    def predict_fine(self, context: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.predictor(context, mask=mask)

    def encode_fine(self, clip: torch.Tensor) -> torch.Tensor:
        return self.predict_fine(self.encode_context(clip))

    def z_from_fine(self, fine: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(fine, self.feature_grid)
        z = self.bottleneck(pooled)
        z = z.permute(0, 2, 3, 1)
        z = self.norm(z)
        return z.permute(0, 3, 1, 2).contiguous()

    def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
        context = self.encode_context(clip)
        fine = self.predict_fine(context)
        z_t = self.z_from_fine(fine)
        h, logits = self.decoder.forward_with_logits(fine)
        close_logits = self.close_head(z_t).squeeze(1)
        return {
            "F_ctx": context,
            "F": fine,
            "Z_t": z_t,
            "Hhat_1s": h,
            "Hhat_1s_logits": logits,
            "Hhat_close": torch.sigmoid(close_logits),
            "Hhat_close_logits": close_logits,
        }

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def encoder_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def predictor_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.predictor.parameters() if p.requires_grad]

    def hint_parameters(self) -> list[nn.Parameter]:
        return self.predictor_parameters()

    def parameter_audit(self) -> dict[str, Any]:
        total = int(sum(p.numel() for p in self.parameters()))
        trainable = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        frozen = int(sum(p.numel() for p in self.parameters() if not p.requires_grad))
        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
            "encoder": int(sum(p.numel() for p in self.encoder.parameters())),
            "predictor": int(sum(p.numel() for p in self.predictor.parameters())),
            "decoder": int(sum(p.numel() for p in self.decoder.parameters())),
            "under_1m": total < MAX_PARAMS,
        }


def load_frozen_teacher_heads(student: CollisionStudent, blob: dict[str, Any]) -> None:
    """Copy G's decoder / bottleneck / LayerNorm / close head and freeze them (eval only)."""
    student.decoder.load_state_dict(blob["decoder"], strict=True)
    adapter = blob["adapter"]
    bot = {k.split("bottleneck.", 1)[1]: v for k, v in adapter.items() if k.startswith("bottleneck.")}
    student.bottleneck.load_state_dict(bot, strict=True)
    norm = {k.split("norm.", 1)[1]: v for k, v in adapter.items() if k.startswith("norm.")}
    student.norm.load_state_dict(norm, strict=True)
    if blob.get("close_head"):
        student.close_head.load_state_dict(blob["close_head"], strict=True)
    student._freeze_teacher_heads()
