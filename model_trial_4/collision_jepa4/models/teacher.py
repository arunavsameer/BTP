"""Collision teacher: frozen V-JEPA plus the 1s-ahead trunk."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .smoother import ConsistencySmoother
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
        self.smoother_enabled = bool(cfg.get("smoother.enabled", True))
        self.smoother = ConsistencySmoother(
            window=int(cfg.get("smoother.window", 4)),
            tau_init=float(cfg.get("smoother.tau_init", 0.5)),
        )
        if not self.smoother_enabled:
            for p in self.smoother.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def reset_smoother(self) -> None:
        self.smoother.reset()

    def encode_tokens(self, clip: torch.Tensor) -> torch.Tensor:
        return self.backbone(clip)

    def forward(
        self,
        clip: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        clip_ids: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if tokens is None:
            if clip is None:
                raise ValueError("forward requires clip or tokens")
            tokens = self.encode_tokens(clip)
        packed = self.trunk(tokens)
        raw_probs = packed["Hhat_1s"]
        raw_logits = packed["Hhat_1s_logits"]
        out = {
            "Hhat_1s_raw": raw_probs,
            "Hhat_1s_raw_logits": raw_logits,
            "logits_fine": packed["logits_fine"],
        }
        if not self.smoother_enabled:
            out["Hhat_1s"] = raw_probs
            out["Hhat_1s_logits"] = raw_logits
            out["alpha"] = raw_probs.new_ones((raw_probs.shape[0],))
            return out
        mixed = self.smoother(raw_probs, raw_logits, clip_ids=clip_ids)
        out["Hhat_1s"] = mixed["Hhat_1s"]
        out["Hhat_1s_logits"] = mixed["Hhat_1s_logits"]
        out["alpha"] = mixed["alpha"]
        return out

    def trunk_parameters(self) -> list[nn.Parameter]:
        return list(self.trunk.parameters())

    def trainable_parameters(self) -> list[nn.Parameter]:
        params = self.trunk_parameters()
        if self.smoother_enabled:
            params.extend(self.smoother.parameters())
        return params

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
            "smoother": int(sum(p.numel() for p in self.smoother.parameters())),
            "smoother_enabled": bool(self.smoother_enabled),
            "smoother_window": int(self.smoother.window),
            "tau": float(self.smoother.tau().detach()),
            "backbone_id": getattr(self.backbone, "model_id", "unknown"),
            "hidden_size": int(self.backbone.hidden_size),
            "temporal_groups": self.temporal_groups,
            "spatial": self.spatial,
            "heads_mode": "direct_h1s",
        }
        if int(self.backbone.hidden_size) == 1408:
            audit["trunk_matches_spec"] = trunk_n == TRUNK_PARAMS_1408
        return audit
