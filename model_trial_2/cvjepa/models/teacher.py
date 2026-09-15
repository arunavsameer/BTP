"""V-JEPA 2 teacher: video tokens → 5×5 collision Z → heatmap.

The giant encoder stays mostly frozen. Phase 1 trains only the spatial adapter,
predictor, and decoder. Phase 2 attaches LoRA to the last encoder blocks so the
latent itself can move toward collision semantics.

Deployment never loads this module — only the tiny student.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import HeatmapDecoder, HeatmapDelta
from .lora import apply_lora_to_last_blocks, lora_parameters


class SpatialAdapter(nn.Module):
    """Tubelet tokens [B, N, C] → collision Z [B, z, grid, grid]."""

    def __init__(self, hidden_size: int, z_channels: int, feature_grid: int = 5, pool_hidden: int = 128):
        super().__init__()
        self.feature_grid = feature_grid
        self.reduce = nn.Sequential(
            nn.Conv2d(hidden_size, pool_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(pool_hidden, pool_hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(pool_hidden, pool_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(pool_hidden, z_channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(z_channels)

    def forward(self, tokens: torch.Tensor, spatial: int) -> torch.Tensor:
        b, n, c = tokens.shape
        t_prime = max(1, n // (spatial * spatial))
        usable = t_prime * spatial * spatial
        tokens = tokens[:, :usable, :]
        grid = tokens.reshape(b, t_prime, spatial, spatial, c)
        grid = grid.mean(dim=1).permute(0, 3, 1, 2).contiguous()
        x = self.reduce(grid)
        x = F.adaptive_avg_pool2d(x, self.feature_grid)
        z = self.bottleneck(x)
        z = z.permute(0, 2, 3, 1)
        z = self.norm(z)
        return z.permute(0, 3, 1, 2).contiguous()


class ResidualPredictor(nn.Module):
    def __init__(self, z_channels: int):
        super().__init__()
        hid = z_channels * 3
        self.net = nn.Sequential(
            nn.Conv2d(z_channels, hid, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, hid, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid, z_channels, kernel_size=1),
        )

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return z_t + self.net(z_t)


class VJEPA2Backbone(nn.Module):
    """Frozen (or LoRA) HuggingFace V-JEPA 2 video encoder."""

    def __init__(
        self,
        hf_model_id: str,
        cache_dir: str | Path | None = None,
        torch_dtype: str = "float16",
    ):
        super().__init__()
        from transformers import AutoModel

        kwargs: dict = {}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[
            str(torch_dtype)
        ]
        kwargs["torch_dtype"] = dtype
        try:
            kwargs["attn_implementation"] = "sdpa"
            self.model = AutoModel.from_pretrained(hf_model_id, **kwargs)
        except Exception:
            kwargs.pop("attn_implementation", None)
            self.model = AutoModel.from_pretrained(hf_model_id, **kwargs)

        cfg = self.model.config
        self.hidden_size = int(getattr(cfg, "hidden_size", 1408))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        crop = getattr(cfg, "crop_size", None)
        if isinstance(crop, dict):
            crop = crop.get("height") or crop.get("width")
        self.crop_size = int(crop or getattr(cfg, "image_size", 256))
        self.tubelet_size = int(getattr(cfg, "tubelet_size", 2))
        mean = getattr(cfg, "image_mean", None) or [0.5, 0.5, 0.5]
        std = getattr(cfg, "image_std", None) or [0.5, 0.5, 0.5]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 1, 3, 1, 1), persistent=False)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self._grad_enabled = False

    @property
    def spatial(self) -> int:
        return max(1, self.crop_size // self.patch_size)

    def enable_lora(self, last_n: int, rank: int, alpha: float) -> int:
        n = apply_lora_to_last_blocks(self.model, last_n=last_n, rank=rank, alpha=alpha)
        self._grad_enabled = True
        try:
            self.model.gradient_checkpointing_enable()
        except Exception:
            pass
        return n

    def set_backbone_grad(self, enabled: bool) -> None:
        self._grad_enabled = bool(enabled)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip [B, T, 3, H, W] in [0, 1] → tokens [B, N, hidden]."""
        x = (clip - self._mean.to(device=clip.device, dtype=clip.dtype)) / self._std.to(
            device=clip.device, dtype=clip.dtype
        )
        ctx = torch.enable_grad() if self._grad_enabled else torch.no_grad()
        with ctx:
            try:
                out = self.model(pixel_values_videos=x)
            except TypeError:
                try:
                    out = self.model.get_vision_features(x)
                    if torch.is_tensor(out):
                        return out.float()
                except Exception:
                    out = self.model(x)
            tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
            return tokens.float()


class CollisionTeacher(nn.Module):
    def __init__(
        self,
        hf_model_id: str,
        z_channels: int = 32,
        feature_grid: int = 5,
        pool_hidden: int = 128,
        cache_dir: str | Path | None = None,
        torch_dtype: str = "float16",
        backbone: VJEPA2Backbone | None = None,
    ):
        super().__init__()
        self.feature_grid = feature_grid
        self.z_channels = z_channels
        if backbone is None:
            self.backbone = VJEPA2Backbone(hf_model_id, cache_dir=cache_dir, torch_dtype=torch_dtype)
        else:
            self.backbone = backbone
        hidden = int(self.backbone.hidden_size)
        self.adapter = SpatialAdapter(hidden, z_channels, feature_grid, pool_hidden)
        self.predictor = ResidualPredictor(z_channels)
        self.decoder = HeatmapDecoder(z_channels=z_channels)
        self.delta_head = HeatmapDelta(z_channels=z_channels)

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(clip)
        return self.adapter(tokens, self.backbone.spatial)

    def _heads(self, z_t: torch.Tensor) -> dict:
        z_plus_hat = self.predictor(z_t)
        h_now = self.decoder(z_t)
        delta = self.delta_head(z_plus_hat - z_t)
        h_plus = (h_now.detach() + delta).clamp(0.0, 1.0)
        return {
            "z_t": z_t,
            "z_plus_hat": z_plus_hat,
            "h_now_hat": h_now,
            "h_plus_hat": h_plus,
            "delta": delta,
        }

    def forward(self, clip: torch.Tensor) -> dict:
        return self._heads(self.encode(clip))

    def predict_future_heatmap(self, clip: torch.Tensor) -> torch.Tensor:
        return self.forward(clip)["h_plus_hat"]

    def adapter_parameters(self) -> list[nn.Parameter]:
        params = list(self.adapter.parameters())
        params += list(self.predictor.parameters())
        params += list(self.decoder.parameters())
        params += list(self.delta_head.parameters())
        return params

    def lora_parameters(self) -> list[nn.Parameter]:
        return lora_parameters(self.backbone)

    def trainable_parameters(self, include_lora: bool = False) -> list[nn.Parameter]:
        params = self.adapter_parameters()
        if include_lora:
            params += self.lora_parameters()
        return params
