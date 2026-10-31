"""Now-only collision trunk on post-LayerNorm V-JEPA tokens.

Token layout is [B, T*H*W, C] with T=8, H=W=16, C=1408 for ViT-G.
Depthwise branches use bias=False. Every other convolution uses bias=True,
which is what the 347,745 parameter count assumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Linear 1408→128 plus two blocks, downsample, fusion, decoder, and the 1×1 head.
TRUNK_PARAMS_1408 = 347_745


def channel_layernorm(norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """LayerNorm over channels at each spatiotemporal location.

    ``x`` is NCHW or NCTHW. ``nn.LayerNorm`` on that layout would normalize the
    last spatial axis, so channels are moved to the end and back.
    """
    y = x.movedim(1, -1)
    y = norm(y)
    return y.movedim(-1, 1).contiguous()


class SpatiotemporalBlock(nn.Module):
    """Two depthwise branches, pointwise mix, per-location LayerNorm, residual."""

    def __init__(self, channels: int = 128):
        super().__init__()
        c = int(channels)
        self.branch_a = nn.Conv3d(
            c, c, kernel_size=(5, 3, 3), padding=(2, 1, 1), groups=c, bias=False
        )
        self.branch_b = nn.Conv3d(
            c, c, kernel_size=(3, 5, 5), padding=(1, 2, 2), groups=c, bias=False
        )
        self.norm_a = nn.LayerNorm(c)
        self.norm_b = nn.LayerNorm(c)
        self.pointwise = nn.Conv3d(c * 2, c, kernel_size=1, bias=True)
        self.norm_out = nn.LayerNorm(c)

    def _branch(self, conv: nn.Conv3d, norm: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
        y = conv(x)
        y = channel_layernorm(norm, y)
        return F.gelu(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.cat(
            [self._branch(self.branch_a, self.norm_a, x), self._branch(self.branch_b, self.norm_b, x)],
            dim=1,
        )
        y = channel_layernorm(self.norm_out, self.pointwise(mixed))
        return x + y


class CollisionTrunk(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1408,
        temporal_groups: int = 8,
        spatial: int = 16,
        channels: int = 128,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.temporal_groups = int(temporal_groups)
        self.spatial = int(spatial)
        self.channels = int(channels)
        if self.temporal_groups != 8 or self.spatial != 16:
            raise ValueError(
                f"trunk expects T=8 and S=16, got T={self.temporal_groups} S={self.spatial}"
            )
        self.projection = nn.Linear(self.hidden_size, self.channels)
        self.blocks = nn.Sequential(SpatiotemporalBlock(self.channels), SpatiotemporalBlock(self.channels))
        self.temporal_downsample = nn.Conv3d(
            self.channels,
            96,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            padding=(1, 0, 0),
            bias=True,
        )
        self.temporal_collapse = nn.Conv3d(96, 48, kernel_size=(4, 1, 1), stride=1, padding=0, bias=True)
        self.temporal_fusion = nn.Conv2d(144, 48, kernel_size=1, bias=True)
        self.decoder = nn.Conv2d(48, 16, kernel_size=3, padding=1, bias=True)
        self.prediction = nn.Conv2d(16, 1, kernel_size=1, bias=True)

    def expected_tokens(self) -> int:
        return self.temporal_groups * self.spatial * self.spatial

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        """[B, N, C] → [B, 128, 8, 16, 16]."""
        b, n, c = tokens.shape
        expected = self.expected_tokens()
        if n != expected:
            raise RuntimeError(
                f"token count {n} != T*S*S = {self.temporal_groups}*{self.spatial}*{self.spatial}={expected}. "
                "Refusing to slice tokens to force a reshape."
            )
        if c != self.hidden_size:
            raise RuntimeError(f"token hidden {c} != trunk hidden_size {self.hidden_size}")
        x = tokens.reshape(b, self.temporal_groups, self.spatial, self.spatial, c)
        x = F.gelu(self.projection(x))
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def aggregate(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 96, 4, 16, 16] → [B, 48, 16, 16].

        Path B is the last downsampled slice (tubelets 5–7, centered on 6),
        not the original last RGB frame.
        """
        if x.shape[2] != 4:
            raise RuntimeError(f"temporal collapse expects T=4, got {tuple(x.shape)}")
        collapsed = self.temporal_collapse(x).squeeze(2)
        current = x[:, :, -1]
        return self.temporal_fusion(torch.cat([collapsed, current], dim=1))

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.blocks(self.project(tokens))
        x = self.temporal_downsample(x)
        x = self.aggregate(x)
        x = F.relu(self.decoder(x))
        logits_fine = self.prediction(x)
        logits_coarse = F.adaptive_max_pool2d(logits_fine, (5, 5))
        return {
            "logits_fine": logits_fine,
            "Hhat_1s_logits": logits_coarse.squeeze(1),
            "Hhat_1s": torch.sigmoid(logits_coarse).squeeze(1),
        }
