"""3D ConvNeXt-V2 (sparse) student: RGB clip → parent F_dec [B,128,16,16]."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .blocks import (
    ConvNeXtV2Block2d,
    ConvNeXtV2Block3d,
    Downsample3d,
    LayerNorm2d,
    LayerNorm3d,
    TemporalSoftmaxPool,
)
from .sparse import apply_submanifold, random_patch_mask, resize_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VARIANTS = {
    "atto": {"depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320], "drop_path": 0.0},
    "tiny": {"depths": [3, 3, 9, 3], "dims": [96, 192, 384, 768], "drop_path": 0.10},
    "base": {"depths": [3, 3, 27, 3], "dims": [128, 256, 512, 1024], "drop_path": 0.20},
    "large": {"depths": [3, 3, 27, 3], "dims": [192, 384, 768, 1536], "drop_path": 0.30},
}


def n_params(mod: nn.Module, *, trainable: bool | None = None) -> int:
    if trainable is None:
        return int(sum(p.numel() for p in mod.parameters()))
    return int(sum(p.numel() for p in mod.parameters() if p.requires_grad is trainable))


def _trunc_normal_(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SparseConvNeXtV2_3D(nn.Module):
    """Video ConvNeXt-V2 with FCMAE-style submanifold sparse convolutions.

    Spatial downsampling is 16× (256→16) so the last map lines up with parent
    ``F_dec``. Temporal path: 16 → 8 (stem) → 4 (last downsample) → softmax pool.
    """

    def __init__(
        self,
        *,
        depths: list[int],
        dims: list[int],
        drop_path: float = 0.3,
        out_channels: int = 128,
        mask_ratio: float = 0.6,
        sparse_prob: float = 0.5,
        grad_checkpoint: bool = True,
        kernel: tuple[int, int, int] = (3, 7, 7),
    ):
        super().__init__()
        if len(depths) != 4 or len(dims) != 4:
            raise ValueError("ConvNeXt-V2 expects 4 stages")
        self.depths = list(depths)
        self.dims = list(dims)
        self.mask_ratio = float(mask_ratio)
        self.sparse_prob = float(sparse_prob)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.out_grid = 16

        self.stem = nn.Sequential(
            nn.Conv3d(3, dims[0], kernel_size=(2, 4, 4), stride=(2, 4, 4)),
            LayerNorm3d(dims[0]),
        )
        dp = torch.linspace(0, float(drop_path), steps=sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        self.downsample = nn.ModuleList()
        strides = [(1, 2, 2), (1, 2, 2), (2, 1, 1)]
        for i in range(4):
            blocks = nn.ModuleList(
                [
                    ConvNeXtV2Block3d(dims[i], drop_path=dp[cur + j], kernel=kernel)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(blocks)
            cur += depths[i]
            if i < 3:
                self.downsample.append(Downsample3d(dims[i], dims[i + 1], strides[i]))

        self.pool = TemporalSoftmaxPool(dims[-1], temporal_groups=4)
        self.mask_token = nn.Parameter(torch.zeros(1, dims[-1], 1, 1))
        self.fcmae = ConvNeXtV2Block2d(dims[-1], drop_path=0.0)
        self.norm = LayerNorm2d(dims[-1])
        self.to_fdec = nn.Conv2d(dims[-1], int(out_channels), kernel_size=1)
        nn.init.zeros_(self.mask_token)
        _trunc_normal_(self)

    def _maybe_mask(self, batch: int, device, dtype) -> torch.Tensor | None:
        if not self.training or self.sparse_prob <= 0 or self.mask_ratio <= 0:
            return None
        if torch.rand((), device=device).item() >= self.sparse_prob:
            return None
        return random_patch_mask(
            batch, self.out_grid, mask_ratio=self.mask_ratio, device=device, dtype=dtype
        )

    def _run_blocks(self, blocks: nn.ModuleList, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        use_ckpt = self.training and self.grad_checkpoint and x.requires_grad

        def _one(blk, z):
            return blk(z, mask)

        for blk in blocks:
            if use_ckpt:
                x = checkpoint(lambda z, b=blk: _one(b, z), x, use_reentrant=False)
            else:
                x = _one(blk, x)
        return x

    def forward_features(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(x)
        if mask is not None:
            mask = resize_mask(mask, x.shape[2:])
            x = apply_submanifold(x, mask)
        x = self._run_blocks(self.stages[0], x, mask)
        for i in range(3):
            x, mask = self.downsample[i](x, mask)
            x = self._run_blocks(self.stages[i + 1], x, mask)
        return x, mask

    def decode_fdec(self, x3d: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        grid = self.pool(x3d)
        if mask is not None:
            m2d = resize_mask(mask, (1, grid.shape[-2], grid.shape[-1])).squeeze(2)
            grid = grid * m2d + (1.0 - m2d) * self.mask_token.to(dtype=grid.dtype)
        grid = self.fcmae(grid)
        return self.to_fdec(self.norm(grid))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = False) -> dict[str, torch.Tensor]:
        """x: [B,3,T,H,W]. ``mask=False`` means sample a training mask; ``None`` is dense."""
        if mask is False:
            mask = self._maybe_mask(x.shape[0], x.device, x.dtype)
        x3d, mask_used = self.forward_features(x, mask)
        f = self.decode_fdec(x3d, mask_used)
        return {"F_3d": x3d, "F": f, "F_ctx": f, "sparse_mask": mask_used}


class CollisionStudentConvNeXtV2(nn.Module):
    def __init__(
        self,
        decoder: nn.Module,
        *,
        variant: str = "large",
        mask_ratio: float = 0.6,
        sparse_prob: float = 0.5,
        grad_checkpoint: bool = True,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}; expected {sorted(VARIANTS)}")
        cfg = VARIANTS[variant]
        self.variant = str(variant)
        self.backbone = SparseConvNeXtV2_3D(
            depths=cfg["depths"],
            dims=cfg["dims"],
            drop_path=cfg["drop_path"],
            mask_ratio=mask_ratio,
            sparse_prob=sparse_prob,
            grad_checkpoint=grad_checkpoint,
        )
        self.decoder = decoder
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.decoder.eval()
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def normalize(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean.to(dtype=clip.dtype, device=clip.device)) / self._std.to(
            dtype=clip.dtype, device=clip.device
        )

    def forward(self, clip: torch.Tensor, mask: torch.Tensor | None = False) -> dict[str, torch.Tensor]:
        x = self.normalize(clip).permute(0, 2, 1, 3, 4).contiguous()
        out = self.backbone(x, mask=mask)
        h, logits = self.decoder.forward_with_logits(out["F"])
        out["Hhat_1s"] = h
        out["Hhat_1s_logits"] = logits
        return out

    def student_parameters(self):
        return (p for n, p in self.named_parameters() if not n.startswith("decoder."))

    def param_audit(self) -> dict:
        return {
            "variant": self.variant,
            "total": n_params(self),
            "trainable": n_params(self, trainable=True),
            "frozen": n_params(self, trainable=False),
            "backbone": n_params(self.backbone),
            "decoder": n_params(self.decoder),
            "depths": self.backbone.depths,
            "dims": self.backbone.dims,
        }
