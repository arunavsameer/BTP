"""Region-Shuffle Collision JEPA student (wearable model).

Training: shuffle the RGB mosaic → encode → unpermute Z → residual predict in
the real k×k layout → heatmap. Encoder cannot lean on a walking-corridor prior;
predictor can still move threat across neighbouring cells.

Deployment never shuffles: RGB clip → Z_t → residual predictor → H(t+1s).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.shuffle import unpermute_map
from .decoder import HeatmapDecoder, HeatmapDelta
from .tiny_cnn import TinyCNN

LOOM_CHANNELS = 4


def pixel_loom(frames: torch.Tensor, grid: int) -> torch.Tensor:
    """Per-cell luminance stats that track expansion / fill-in.

    Returns [B, 4, grid, grid]: mean now, mean growth, max growth, fill (max−mean).
    """
    b, n, _, h, w = frames.shape
    lum = 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]
    lum = lum.reshape(b * n, 1, h, w)
    avg = F.adaptive_avg_pool2d(lum, (grid, grid)).reshape(b, n, 1, grid, grid)
    mx = F.adaptive_max_pool2d(lum, (grid, grid)).reshape(b, n, 1, grid, grid)
    avg_n = avg[:, -1]
    mx_n = mx[:, -1]
    return torch.cat([avg_n, avg_n - avg[:, 0], mx_n - mx[:, 0], mx_n - avg_n], dim=1)


class CollisionEncoder(nn.Module):
    """Motion stack → Z_t, then per-cell LayerNorm so JEPA is scale-stable."""

    def __init__(self, in_channels: int, feat_channels: int, z_channels: int):
        super().__init__()
        c = feat_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1),
            nn.GroupNorm(8 if c >= 8 else 1, c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 2, z_channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(z_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = z.permute(0, 2, 3, 1)
        z = self.norm(z)
        return z.permute(0, 3, 1, 2).contiguous()


class ResidualPredictor(nn.Module):
    """Zhat+ = Z_t + P(Z_t). Dilated 3×3 lets threat move across neighbouring cells."""

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


class ContextEncoder(nn.Module):
    """Shared TinyCNN + collision encoder. This is what the EMA teacher copies."""

    def __init__(
        self,
        width: int,
        feature_grid: int,
        z_channels: int,
        n_frames: int,
        use_loom: bool = True,
    ):
        super().__init__()
        self.feature_grid = feature_grid
        self.n_frames = n_frames
        self.use_loom = use_loom
        self.cnn = TinyCNN(width=width, feature_grid=feature_grid)
        in_ch = self.cnn.feat_channels * n_frames
        if use_loom:
            in_ch += LOOM_CHANNELS
        self.encoder = CollisionEncoder(in_ch, self.cnn.feat_channels, z_channels)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = frames.shape
        if n != self.n_frames:
            raise ValueError(f"expected {self.n_frames} frames, got {n}")
        feats = self.cnn(frames.reshape(b * n, c, h, w))
        feats = feats.reshape(b, n, feats.shape[1], feats.shape[2], feats.shape[3])
        now = feats[:, -1]
        parts = [now]
        for i in range(n - 1):
            parts.append(now - feats[:, i])
        x = torch.cat(parts, dim=1)
        if self.use_loom:
            x = torch.cat([x, pixel_loom(frames, self.feature_grid)], dim=1)
        return self.encoder(x)


class RSJEPA(nn.Module):
    def __init__(
        self,
        width: int = 32,
        feature_grid: int = 5,
        z_channels: int = 16,
        ema_momentum: float = 0.996,
        n_frames: int = 3,
        use_loom: bool = True,
        copy_residual: bool = False,
    ):
        super().__init__()
        self.feature_grid = feature_grid
        self.ema_momentum = ema_momentum
        self.n_frames = n_frames
        self.copy_residual = copy_residual
        self.context = ContextEncoder(width, feature_grid, z_channels, n_frames, use_loom)
        self.predictor = ResidualPredictor(z_channels)
        self.decoder = HeatmapDecoder(z_channels=z_channels)
        self.delta_head = HeatmapDelta(z_channels=z_channels) if copy_residual else None
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @classmethod
    def from_config(cls, cfg) -> "RSJEPA":
        offsets = list(cfg.get("student.frame_offsets"))
        return cls(
            width=int(cfg.get("student.cnn_width", 32)),
            feature_grid=int(cfg.get("student.feature_grid", 5)),
            z_channels=int(cfg.get("student.z_channels", 16)),
            ema_momentum=float(cfg.get("jepa.ema_momentum", 0.996)),
            n_frames=len(offsets),
            use_loom=bool(cfg.get("student.use_loom", True)),
            copy_residual=bool(cfg.get("student.copy_residual", False)),
        )

    def _align(self, z: torch.Tensor, perm: torch.Tensor | None) -> torch.Tensor:
        if perm is None:
            return z
        return unpermute_map(z, perm, self.feature_grid)

    def _future_from_now(self, z_t: torch.Tensor, z_plus_hat: torch.Tensor) -> dict:
        h_now_hat = self.decoder(z_t)
        if self.copy_residual:
            delta = self.delta_head(z_plus_hat - z_t)
            base = h_now_hat.detach()
            h_plus_hat = (base + delta).clamp(0.0, 1.0)
            h_mid_hat = (base + 0.5 * delta).clamp(0.0, 1.0)
        else:
            delta = None
            h_plus_hat = self.decoder(z_plus_hat)
            h_mid_hat = self.decoder(z_t + 0.5 * (z_plus_hat - z_t))
        return {
            "h_now_hat": h_now_hat,
            "h_plus_hat": h_plus_hat,
            "h_mid_hat": h_mid_hat,
            "delta": delta,
        }

    def forward(
        self,
        frames: torch.Tensor,
        frames_future: torch.Tensor | None = None,
        perm: torch.Tensor | None = None,
    ) -> dict:
        z_t = self._align(self.context(frames), perm)
        z_plus_hat = self.predictor(z_t)
        out = {"z_t": z_t, "z_plus_hat": z_plus_hat, **self._future_from_now(z_t, z_plus_hat)}
        if frames_future is not None:
            with torch.no_grad():
                self.target.eval()
                out["z_plus"] = self._align(self.target(frames_future), perm)
        return out

    def predict_future_heatmap(self, frames: torch.Tensor) -> torch.Tensor:
        z_t = self.context(frames)
        z_plus_hat = self.predictor(z_t)
        return self._future_from_now(z_t, z_plus_hat)["h_plus_hat"]

    @torch.no_grad()
    def update_ema(self) -> None:
        m = self.ema_momentum
        for p_t, p_s in zip(self.target.parameters(), self.context.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
        for b_t, b_s in zip(self.target.buffers(), self.context.buffers()):
            b_t.data.copy_(b_s.data)
