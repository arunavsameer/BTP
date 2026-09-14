"""Teacher: frozen V-JEPA-2 + learned spatial pool + bottleneck -> Z (5x5x16).

Only ever run offline (Stage A training and Stage B caching). The V-JEPA-2 backbone
is frozen; we only train the small pool/bottleneck and the shared heatmap decoder.

The backbone is loaded lazily via HuggingFace so that the rest of the project (data,
baselines, student) does not require downloading multi-GB weights.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .decoder import HeatmapDecoder


class VJEPA2Backbone(nn.Module):
    """Thin frozen wrapper around a HuggingFace V-JEPA-2 video encoder."""

    def __init__(self, hf_model_id: str, fp16: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(hf_model_id)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = self.model.config
        self.hidden_size = int(getattr(cfg, "hidden_size", 1024))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.tubelet_size = int(getattr(cfg, "tubelet_size", 2))
        self.crop_size = int(getattr(cfg, "crop_size", getattr(cfg, "image_size", 256)))
        # Normalization stats (fall back to 0.5 as V-JEPA-2 uses).
        mean = getattr(cfg, "image_mean", None) or [0.5, 0.5, 0.5]
        std = getattr(cfg, "image_std", None) or [0.5, 0.5, 0.5]
        self.register_buffer("_mean", torch.tensor(mean).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(std).view(1, 1, 3, 1, 1), persistent=False)
        self._fp16 = fp16

    @torch.no_grad()
    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """clip: [B, T, 3, H, W] in [0, 1]. Returns tokens [B, N, hidden]."""
        clip = (clip - self._mean.to(clip.device)) / self._std.to(clip.device)
        if self._fp16 and clip.is_cuda:
            clip = clip.half()
            self.model.half()
        # HuggingFace V-JEPA-2 expects `pixel_values_videos` [B, T, C, H, W].
        try:
            out = self.model(pixel_values_videos=clip)
        except TypeError:
            out = self.model(clip)
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return tokens.float()

    def token_grid(self, n_tokens: int) -> tuple[int, int, int]:
        """Infer (t', h', w') from the token count and the crop/patch sizes."""
        spatial = self.crop_size // self.patch_size
        t_prime = max(1, n_tokens // (spatial * spatial))
        return t_prime, spatial, spatial


class CollisionTeacher(nn.Module):
    def __init__(
        self,
        hf_model_id: str,
        z_channels: int = 16,
        feature_grid: int = 5,
        pool_hidden: int = 64,
        fp16: bool = True,
    ):
        super().__init__()
        self.backbone = VJEPA2Backbone(hf_model_id, fp16=fp16)
        self.feature_grid = feature_grid
        h = self.backbone.hidden_size

        # Trainable adapter: reduce channels -> pool to grid -> bottleneck to Z.
        self.reduce = nn.Sequential(
            nn.Conv2d(h, pool_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(feature_grid)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(pool_hidden, pool_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(pool_hidden, z_channels, kernel_size=1),
        )
        self.decoder = HeatmapDecoder(z_channels=z_channels)

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        """clip -> Z [B, z_channels, grid, grid]."""
        tokens = self.backbone(clip)  # [B, N, hidden]
        b, n, c = tokens.shape
        t_prime, h_prime, w_prime = self.backbone.token_grid(n)
        usable = t_prime * h_prime * w_prime
        tokens = tokens[:, :usable, :]
        grid = tokens.reshape(b, t_prime, h_prime, w_prime, c)
        grid = grid.mean(dim=1)  # average over temporal tokens -> [B, h', w', c]
        grid = grid.permute(0, 3, 1, 2).contiguous()  # [B, c, h', w']
        x = self.reduce(grid)
        x = self.pool(x)
        z = self.bottleneck(x)
        return z

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(clip)
        h_hat = self.decoder(z)
        return z, h_hat

    def trainable_parameters(self):
        params = list(self.reduce.parameters())
        params += list(self.bottleneck.parameters())
        params += list(self.decoder.parameters())
        return params
