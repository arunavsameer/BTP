"""Teacher: frozen V-JEPA-2 + learned spatial pool + bottleneck -> Z (5x5x16).

Only ever run offline (Stage A training and Stage B caching). The V-JEPA-2 backbone
is frozen; we only train the small pool/bottleneck and the shared heatmap decoder.

The backbone is loaded lazily via HuggingFace so that the rest of the project (data,
baselines, student) does not require downloading multi-GB weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import HeatmapDecoder, OccupancyHead


class VJEPA2Backbone(nn.Module):
    """Thin frozen wrapper around a HuggingFace V-JEPA-2 video encoder."""

    def __init__(self, hf_model_id: str, fp16: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(hf_model_id, attn_implementation="sdpa")
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Encoder tokens are all we use. The JEPA predictor is 12 extra layers
        # and is on by default — drop it so it never sits on the GPU.
        if hasattr(self.model, "predictor"):
            del self.model.predictor

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
        if self._fp16 and clip.is_cuda and clip.dtype != torch.float16:
            clip = clip.half()
        # HuggingFace V-JEPA-2 expects `pixel_values_videos` [B, T, C, H, W].
        # Call the encoder only. Full VJEPA2Model.forward also runs the predictor
        # unless skip_predictor=True, and still builds unused mask gathers.
        encoder = getattr(self.model, "encoder", None)
        if encoder is not None:
            out = encoder(pixel_values_videos=clip)
        else:
            try:
                out = self.model(pixel_values_videos=clip, skip_predictor=True)
            except TypeError:
                out = self.model(clip)
        tokens = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return tokens.float()

    def token_grid(self, n_tokens: int) -> tuple[int, int, int]:
        """Infer (t', h', w') from the token count and the crop/patch sizes."""
        spatial = self.crop_size // self.patch_size
        t_prime = max(1, n_tokens // (spatial * spatial))
        return t_prime, spatial, spatial


class SpatialAttentionPool(nn.Module):
    """5x5 learned queries over the backbone spatial tokens.

    Avg-pool mixes a person with leaves in the same coarse cell. Attention lets
    the collision query pick the threat token and ignore texture.
    """

    def __init__(self, channels: int, grid: int = 5):
        super().__init__()
        self.grid = grid
        self.queries = nn.Parameter(torch.randn(1, grid * grid, channels) * 0.02)
        self.scale = channels ** -0.5
        self.to_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, _, _ = x.shape
        k = self.to_k(x).flatten(2)  # [B, C, HW]
        v = self.to_v(x).flatten(2)
        q = self.queries.expand(b, -1, -1)  # [B, 25, C]
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = attn @ v.transpose(1, 2)  # [B, 25, C]
        return out.transpose(1, 2).reshape(b, c, self.grid, self.grid)


class CollisionTeacher(nn.Module):
    def __init__(
        self,
        hf_model_id: str,
        z_channels: int = 16,
        feature_grid: int = 5,
        pool_hidden: int = 64,
        occ_channels: int = 4,
        fp16: bool = True,
    ):
        super().__init__()
        self.backbone = VJEPA2Backbone(hf_model_id, fp16=fp16)
        self.feature_grid = feature_grid
        h = self.backbone.hidden_size

        # Trainable adapter: reduce channels -> attend to grid -> bottleneck to Z.
        self.reduce = nn.Sequential(
            nn.Conv2d(h, pool_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        # Mix last temporal token (approach) with the mean (context).
        self.temporal_last_logit = nn.Parameter(torch.tensor(1.0))
        self.pool = SpatialAttentionPool(pool_hidden, grid=feature_grid)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(pool_hidden, pool_hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(pool_hidden, z_channels, kernel_size=1),
        )
        self.decoder = HeatmapDecoder(z_channels=z_channels)
        self.occupancy = OccupancyHead(occ_channels=occ_channels)

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        """clip -> Z [B, z_channels, grid, grid]."""
        tokens = self.backbone(clip)  # [B, N, hidden]
        b, n, c = tokens.shape
        t_prime, h_prime, w_prime = self.backbone.token_grid(n)
        usable = t_prime * h_prime * w_prime
        tokens = tokens[:, :usable, :]
        grid = tokens.reshape(b, t_prime, h_prime, w_prime, c)
        last = grid[:, -1]
        mean = grid.mean(dim=1)
        w_last = torch.sigmoid(self.temporal_last_logit)
        mixed = w_last * last + (1.0 - w_last) * mean
        feat = mixed.permute(0, 3, 1, 2).contiguous()  # [B, c, h', w']
        x = self.reduce(feat)
        x = self.pool(x)
        z = self.bottleneck(x)
        return z

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(clip)
        h_hat = self.decoder(z)
        occ_logits = self.occupancy.logits(z)
        return z, h_hat, occ_logits

    def trainable_parameters(self):
        params = list(self.reduce.parameters())
        params += [self.temporal_last_logit]
        params += list(self.pool.parameters())
        params += list(self.bottleneck.parameters())
        params += list(self.decoder.parameters())
        params += list(self.occupancy.parameters())
        return params
