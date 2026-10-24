"""Collision teacher: frozen/LoRA V-JEPA + spatial adapter + multi-horizon residuals."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import heads_mode
from .heatmap_decoder import HeatmapDecoder, HeatmapDelta, copy_residual_future
from .lora import lora_parameters
from .residual_predictor import ResidualPredictor
from .spatial_adapter import SpatialTemporalAdapter
from .vjepa_backbone import DummyVJEPA2Backbone, VJEPA2Backbone, build_backbone


class CollisionTeacher3(nn.Module):
    def __init__(
        self,
        cfg,
        backbone: VJEPA2Backbone | DummyVJEPA2Backbone | None = None,
        dummy_backbone: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.z_channels = int(cfg.get("adapter.z_channels", 32))
        self.feature_grid = int(cfg.get("adapter.feature_grid", 5))
        self.horizon_names = list(cfg.get("sampling.horizon_names", ["1s", "1_5s", "2s"]))
        if backbone is None:
            backbone = build_backbone(cfg, dummy=dummy_backbone)
        self.backbone = backbone
        num_frames = int(cfg.get("sampling.input_frames", 16))
        if hasattr(backbone, "expected_tokens") and callable(getattr(backbone, "expected_tokens")):
            try:
                t_prime, spatial, _ = backbone.expected_tokens(num_frames)
            except TypeError:
                t_prime = getattr(backbone, "temporal_groups")
                spatial = getattr(backbone, "spatial")
        else:
            t_prime = getattr(backbone, "temporal_groups")
            spatial = getattr(backbone, "spatial")
        self.temporal_groups = int(t_prime)
        self.spatial = int(spatial)
        dilations = cfg.get("adapter.temporal_dilations", None)
        decoder_dilated = bool(cfg.get("adapter.decoder_dilated", True))
        self.decode_from = str(cfg.get("adapter.decode_from", "fine")).lower()
        if self.decode_from not in {"fine", "z"}:
            raise ValueError(f"unsupported adapter.decode_from {self.decode_from}")
        self.heatmap_grid = int(cfg.get("adapter.heatmap_grid", 16))
        heatmap_pool = str(cfg.get("adapter.heatmap_pool", "max"))
        proj_c = int(cfg.get("adapter.projection_channels", 128))
        dec_ch = proj_c if self.decode_from == "fine" else self.z_channels
        self.adapter = SpatialTemporalAdapter(
            hidden_size=int(backbone.hidden_size),
            temporal_groups=self.temporal_groups,
            spatial=self.spatial,
            z_channels=self.z_channels,
            feature_grid=self.feature_grid,
            projection_channels=int(cfg.get("adapter.projection_channels", 128)),
            temporal_blocks=int(cfg.get("adapter.temporal_blocks", 4)),
            temporal_kernel=int(cfg.get("adapter.temporal_kernel", 3)),
            temporal_dilation=int(cfg.get("adapter.temporal_dilation", 1)),
            temporal_dilations=dilations,
            temporal_pool=str(cfg.get("adapter.temporal_pool", "softmax")),
            group_norm_groups=int(cfg.get("adapter.group_norm_groups", 8)),
        )
        self.decoder = HeatmapDecoder(
            z_channels=dec_ch,
            dilated=decoder_dilated,
            out_grid=self.feature_grid,
            pool=heatmap_pool,
        )
        self.direct_h1s = heads_mode(cfg) == "direct_h1s"
        if self.direct_h1s:
            self.predictors = nn.ModuleDict()
            self.delta_heads = nn.ModuleDict()
            self.predictor_1s = None
            self.predictor_1_5s = None
            self.predictor_2s = None
        else:
            self.predictors = nn.ModuleDict(
                {name: ResidualPredictor(self.z_channels) for name in self.horizon_names}
            )
            self.delta_heads = nn.ModuleDict(
                {name: HeatmapDelta(self.z_channels, dilated=decoder_dilated) for name in self.horizon_names}
            )
            self.predictor_1s = self.predictors["1s"]
            self.predictor_1_5s = self.predictors["1_5s"]
            self.predictor_2s = self.predictors["2s"]

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_tokens(self, clip: torch.Tensor) -> torch.Tensor:
        return self.backbone(clip)

    def encode_pair(
        self, clip: torch.Tensor | None = None, tokens: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens is None:
            if clip is None:
                raise ValueError("encode requires clip or tokens")
            tokens = self.encode_tokens(clip)
        return self.adapter.features(tokens)

    def encode(self, clip: torch.Tensor | None = None, tokens: torch.Tensor | None = None) -> torch.Tensor:
        z_t, _ = self.encode_pair(clip=clip, tokens=tokens)
        return z_t

    def _heatmap_feat(self, z_t: torch.Tensor, fine: torch.Tensor | None) -> torch.Tensor:
        if self.decode_from != "fine" or fine is None:
            return z_t
        feat = fine
        g = self.heatmap_grid
        if g > 0 and (feat.shape[-2] != g or feat.shape[-1] != g):
            feat = F.interpolate(feat, size=(g, g), mode="bilinear", align_corners=False)
        return feat

    def _heads(self, z_t: torch.Tensor, fine: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        feat = self._heatmap_feat(z_t, fine)
        h_now, logits_now = self.decoder.forward_with_logits(feat)
        if self.direct_h1s:
            return {"Z_t": z_t, "Hhat_1s": h_now, "Hhat_1s_logits": logits_now}
        out: dict[str, torch.Tensor] = {
            "Z_t": z_t,
            "Hhat_now": h_now,
            "Hhat_now_logits": logits_now,
        }
        for name in self.horizon_names:
            z_hat = self.predictors[name](z_t)
            delta = self.delta_heads[name](z_hat - z_t)
            h_hat = copy_residual_future(h_now, delta)
            out[f"Zhat_{name}"] = z_hat
            out[f"Delta_{name}"] = delta
            out[f"Hhat_{name}"] = h_hat
        return out

    def forward(self, clip: torch.Tensor | None = None, tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_t, fine = self.encode_pair(clip=clip, tokens=tokens)
        return self._heads(z_t, fine)

    def adapter_parameters(self) -> list[nn.Parameter]:
        params = list(self.adapter.parameters())
        params += list(self.decoder.parameters())
        params += list(self.predictors.parameters())
        params += list(self.delta_heads.parameters())
        return params

    def lora_parameters(self) -> list[nn.Parameter]:
        if hasattr(self.backbone, "lora_parameters"):
            return self.backbone.lora_parameters()
        return lora_parameters(self.backbone)

    def trainable_parameters(self, include_lora: bool = False) -> list[nn.Parameter]:
        params = self.adapter_parameters()
        if include_lora:
            params += self.lora_parameters()
        return params

    def enable_lora(self) -> int:
        cfg = self.cfg
        n = self.backbone.enable_lora(
            last_n=int(cfg.get("lora.last_n_blocks", 6)),
            rank=int(cfg.get("lora.rank", 16)),
            alpha=float(cfg.get("lora.alpha", 16)),
            dropout=float(cfg.get("lora.dropout", 0.0)),
        )
        if bool(cfg.get("backbone.gradient_checkpointing", True)):
            try:
                self.backbone.model.gradient_checkpointing_enable()
            except Exception:
                pass
        return n

    def parameter_audit(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = [p for p in self.parameters() if p.requires_grad]
        frozen = [p for p in self.parameters() if not p.requires_grad]
        lora = self.lora_parameters()
        adapter = self.adapter_parameters()
        return {
            "total": int(total),
            "trainable": int(sum(p.numel() for p in trainable)),
            "frozen": int(sum(p.numel() for p in frozen)),
            "adapter": int(sum(p.numel() for p in adapter)),
            "lora": int(sum(p.numel() for p in lora)),
            "lora_tensors": len(lora),
            "backbone_id": getattr(self.backbone, "model_id", "unknown"),
            "hidden_size": int(self.backbone.hidden_size),
            "temporal_groups": self.temporal_groups,
            "spatial": self.spatial,
            "z_shape": [self.z_channels, self.feature_grid, self.feature_grid],
            "heads_mode": "direct_h1s" if self.direct_h1s else "residual",
            "decode_from": self.decode_from,
            "heatmap_grid": self.heatmap_grid,
        }
