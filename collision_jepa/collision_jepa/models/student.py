"""Student: the actual wearable model.

3 cheap frames -> shared tiny CNN -> feature differences -> current collision state
Z_t -> residual predictor -> future Zhat+ -> (frozen teacher) decoder -> Hhat+.

Kept small (~1M params target) and free of any V-JEPA / tracker / optical flow.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import HeatmapDecoder
from .tiny_cnn import TinyCNN


class CollisionEncoder(nn.Module):
    """[F3, D1, D2] -> Z_t (grid x grid x z_channels)."""

    def __init__(self, feat_channels: int, z_channels: int):
        super().__init__()
        c = feat_channels
        self.net = nn.Sequential(
            nn.Conv2d(c * 3, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, z_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualPredictor(nn.Module):
    """Predict the CHANGE in collision state: Zhat+ = Z_t + P(Z_t, D1, D2, tau).

    Predicting the residual (not the whole future Z) stops the model from lazily
    copying the current state, which the corridor prior otherwise rewards.
    """

    def __init__(self, z_channels: int, feat_channels: int, diff_dim: int = 32, use_tau: bool = False):
        super().__init__()
        self.use_tau = use_tau
        self.reduce_d = nn.Conv2d(feat_channels, diff_dim, kernel_size=1)
        in_ch = z_channels + 2 * diff_dim + (1 if use_tau else 0)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, z_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(z_channels * 2, z_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(z_channels * 2, z_channels, kernel_size=1),
        )

    def forward(self, z_t, d1, d2, tau=None) -> torch.Tensor:
        d1r = self.reduce_d(d1)
        d2r = self.reduce_d(d2)
        parts = [z_t, d1r, d2r]
        if self.use_tau:
            b, _, h, w = z_t.shape
            tau_map = torch.full((b, 1, h, w), float(tau if tau is not None else 1.0), device=z_t.device)
            parts.append(tau_map)
        delta = self.net(torch.cat(parts, dim=1))
        return z_t + delta


class Student(nn.Module):
    def __init__(
        self,
        width: int = 32,
        feature_grid: int = 5,
        z_channels: int = 16,
        use_tau: bool = False,
    ):
        super().__init__()
        self.cnn = TinyCNN(width=width, feature_grid=feature_grid)
        c = self.cnn.feat_channels
        self.encoder = CollisionEncoder(c, z_channels)
        self.predictor = ResidualPredictor(z_channels, c, use_tau=use_tau)
        self.decoder = HeatmapDecoder(z_channels=z_channels)

    def _features(self, frames: torch.Tensor):
        b, n, c, h, w = frames.shape
        feats = self.cnn(frames.reshape(b * n, c, h, w))
        feats = feats.reshape(b, n, feats.shape[1], feats.shape[2], feats.shape[3])
        f1, f2, f3 = feats[:, 0], feats[:, 1], feats[:, 2]
        d1 = f3 - f2
        d2 = f3 - f1
        return f3, d1, d2

    def forward(self, frames: torch.Tensor, tau=None) -> dict:
        """Full training forward. Returns z_t, z_plus_hat, h_plus_hat, h_now_hat."""
        f3, d1, d2 = self._features(frames)
        z_t = self.encoder(torch.cat([f3, d1, d2], dim=1))
        z_plus_hat = self.predictor(z_t, d1, d2, tau)
        h_plus_hat = self.decoder(z_plus_hat)
        h_now_hat = self.decoder(z_t)
        return {
            "z_t": z_t,
            "z_plus_hat": z_plus_hat,
            "h_plus_hat": h_plus_hat,
            "h_now_hat": h_now_hat,
        }

    def predict_future_heatmap(self, frames: torch.Tensor, tau=None) -> torch.Tensor:
        """Deployment path: frames -> future 5x5 heatmap."""
        f3, d1, d2 = self._features(frames)
        z_t = self.encoder(torch.cat([f3, d1, d2], dim=1))
        z_plus_hat = self.predictor(z_t, d1, d2, tau)
        return self.decoder(z_plus_hat)

    def load_frozen_decoder(self, decoder_state: dict) -> None:
        """Load the teacher decoder weights and freeze them."""
        self.decoder.load_state_dict(decoder_state)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        params = list(self.cnn.parameters())
        params += list(self.encoder.parameters())
        params += list(self.predictor.parameters())
        # decoder intentionally excluded when frozen
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        return params
