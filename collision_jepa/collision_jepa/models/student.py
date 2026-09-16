"""Student: the actual wearable model.

3 cheap frames -> TinyCNN -> Z_t and Zhat+ = Z_t + P(Z_t, motion).
Future heatmap is residual in latent space:

  Ĥ_t  = decoder(Z_t)
  ΔH   = head(Zhat+ − Z_t)
  Ĥ+   = clamp(sg(Ĥ_t) + ΔH ⊙ occupancy(Zhat+))

decoder(Zhat+) is an auxiliary readout so the JEPA predictor is not a dead branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import HeatmapDecoder, OccupancyHead
from .tiny_cnn import TinyCNN


class TemporalConv3(nn.Module):
    """Order-aware motion from F1, F2, F3 via a depthwise temporal 3-tap.

    Input frames are stacked as [B, C, T=3, grid, grid]. A Conv3d with kernel
    (3, 1, 1) sees LEFT→CENTER→CENTER vs LEFT→LEFT→LEFT, which two subtractions
    cannot distinguish. Output is a single [B, C, grid, grid] motion map.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw = nn.Conv3d(
            channels, channels, kernel_size=(3, 1, 1), padding=0, groups=channels, bias=False
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor) -> torch.Tensor:
        x = torch.stack([f1, f2, f3], dim=2)  # [B, C, 3, H, W]
        x = self.dw(x).squeeze(2)
        return self.act(self.bn(self.pw(x)))


class CollisionEncoder(nn.Module):
    """Fused current+motion features -> Z_t (grid x grid x z_channels)."""

    def __init__(self, in_channels: int, z_channels: int):
        super().__init__()
        c = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, z_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualPredictor(nn.Module):
    """JEPA predictor: Zhat+ = Z_t + P(Z_t, motion)."""

    def __init__(self, z_channels: int, motion_channels: int, diff_dim: int = 32):
        super().__init__()
        self.reduce = nn.Conv2d(motion_channels, diff_dim, kernel_size=1)
        in_ch = z_channels + diff_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, z_channels * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(z_channels * 2, z_channels, kernel_size=1),
        )

    def forward(self, z_t: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z_t, self.reduce(motion)], dim=1))
        return z_t + delta


class DeltaHeatmapHead(nn.Module):
    """Map latent residual (Zhat+ − Z_t) to heatmap change in [-1, 1]."""

    def __init__(self, in_channels: int):
        super().__init__()
        hidden = max(in_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


class Student(nn.Module):
    def __init__(
        self,
        width: int = 32,
        feature_grid: int = 5,
        z_channels: int = 16,
        occ_channels: int = 4,
        use_tau: bool = False,
    ):
        super().__init__()
        del use_tau  # kept in the signature so old call sites still work
        self.cnn = TinyCNN(width=width, feature_grid=feature_grid)
        c = self.cnn.feat_channels
        self.temporal = TemporalConv3(c)
        self.fuse = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.encoder = CollisionEncoder(c, z_channels)
        self.predictor = ResidualPredictor(z_channels, c)
        self.decoder = HeatmapDecoder(z_channels=z_channels)
        self.occupancy = OccupancyHead(occ_channels=occ_channels)
        self.delta_h = DeltaHeatmapHead(z_channels)

    def _features(self, frames: torch.Tensor):
        b, n, c, h, w = frames.shape
        if n != 3:
            raise ValueError(f"Student expects 3 frames, got {n}")
        feats = self.cnn(frames.reshape(b * n, c, h, w))
        feats = feats.reshape(b, n, feats.shape[1], feats.shape[2], feats.shape[3])
        f1, f2, f3 = feats[:, 0], feats[:, 1], feats[:, 2]
        motion = self.temporal(f1, f2, f3)
        fused = self.fuse(torch.cat([f3, motion], dim=1))
        return f3, motion, fused

    def forward(self, frames: torch.Tensor, tau=None) -> dict:
        """Ĥ+ comes from Zhat+ − Z_t, not from RGB fused features alone."""
        del tau
        _f3, motion, fused = self._features(frames)
        z_t = self.encoder(fused)
        z_plus_hat = self.predictor(z_t, motion)
        h_now_hat = self.decoder(z_t)
        dh_hat = self.delta_h(z_plus_hat - z_t)
        occ_plus = self.occupancy.prob(z_plus_hat)
        occ_now_logits = self.occupancy.logits(z_t)
        occ_plus_logits = self.occupancy.logits(z_plus_hat)
        # Occupancy gate: leaves (occ≈0) cannot raise STOP via ΔH.
        h_plus_hat = torch.clamp(h_now_hat.detach() + dh_hat * occ_plus, 0.0, 1.0)
        h_plus_latent = self.decoder(z_plus_hat)
        return {
            "z_t": z_t,
            "z_plus_hat": z_plus_hat,
            "h_now_hat": h_now_hat,
            "dh_hat": dh_hat,
            "h_plus_hat": h_plus_hat,
            "h_plus_latent": h_plus_latent,
            "occ_plus": occ_plus,
            "occ_now_logits": occ_now_logits,
            "occ_plus_logits": occ_plus_logits,
        }

    def predict_future_heatmap(self, frames: torch.Tensor, tau=None) -> torch.Tensor:
        """Deployment path: frames -> future 5x5 heatmap."""
        return self.forward(frames, tau)["h_plus_hat"]

    def load_frozen_decoder(self, decoder_state: dict) -> None:
        """Load the teacher decoder weights and freeze them."""
        self.decoder.load_state_dict(decoder_state)
        for p in self.decoder.parameters():
            p.requires_grad_(False)

    def load_frozen_occupancy(self, occupancy_state: dict) -> None:
        self.occupancy.load_state_dict(occupancy_state)
        for p in self.occupancy.parameters():
            p.requires_grad_(False)

    def trainable_parameters(self):
        params = list(self.cnn.parameters())
        params += list(self.temporal.parameters())
        params += list(self.fuse.parameters())
        params += list(self.encoder.parameters())
        params += list(self.predictor.parameters())
        params += list(self.delta_h.parameters())
        params += [p for p in self.decoder.parameters() if p.requires_grad]
        params += [p for p in self.occupancy.parameters() if p.requires_grad]
        return params
