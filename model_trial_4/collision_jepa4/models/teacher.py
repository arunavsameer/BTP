"""Collision teacher: frozen V-JEPA plus the now-only trunk."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .trunk import CollisionTrunk, TRUNK_PARAMS_1408
from .vjepa_backbone import DummyVJEPA2Backbone, VJEPA2Backbone, build_backbone


class CollisionTeacher4(nn.Module):
    def __init__(
        self,
        cfg,
        backbone: VJEPA2Backbone | DummyVJEPA2Backbone | None = None,
        dummy_backbone: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        if backbone is None:
            backbone = build_backbone(cfg, dummy=dummy_backbone)
        self.backbone = backbone
        num_frames = int(cfg.get("sampling.input_frames", 16))
        if hasattr(backbone, "expected_tokens") and callable(getattr(backbone, "expected_tokens")):
            t_prime, spatial, _ = backbone.expected_tokens(num_frames)
        else:
            t_prime = getattr(backbone, "temporal_groups")
            spatial = getattr(backbone, "spatial")
        self.temporal_groups = int(t_prime)
        self.spatial = int(spatial)
        self.trunk = CollisionTrunk(
            hidden_size=int(backbone.hidden_size),
            temporal_groups=self.temporal_groups,
            spatial=self.spatial,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode_tokens(self, clip: torch.Tensor) -> torch.Tensor:
        return self.backbone(clip)

    def forward(
        self, clip: torch.Tensor | None = None, tokens: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if tokens is None:
            if clip is None:
                raise ValueError("forward requires clip or tokens")
            tokens = self.encode_tokens(clip)
        packed = self.trunk(tokens)
        return {
            "Hhat_now": packed["Hhat_now"],
            "Hhat_now_logits": packed["Hhat_now_logits"],
        }

    def trunk_parameters(self) -> list[nn.Parameter]:
        return list(self.trunk.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        return self.trunk_parameters()

    def parameter_audit(self) -> dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = [p for p in self.parameters() if p.requires_grad]
        frozen = [p for p in self.parameters() if not p.requires_grad]
        trunk = self.trunk_parameters()
        trunk_n = int(sum(p.numel() for p in trunk))
        audit: dict[str, Any] = {
            "total": int(total),
            "trainable": int(sum(p.numel() for p in trainable)),
            "frozen": int(sum(p.numel() for p in frozen)),
            "trunk": trunk_n,
            "backbone_id": getattr(self.backbone, "model_id", "unknown"),
            "hidden_size": int(self.backbone.hidden_size),
            "temporal_groups": self.temporal_groups,
            "spatial": self.spatial,
            "heads_mode": "now",
        }
        if int(self.backbone.hidden_size) == 1408:
            audit["trunk_matches_spec"] = trunk_n == TRUNK_PARAMS_1408
        return audit
