"""1s-ahead BCE-with-logits. Later horizons stay in the batch and are not scored."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .heatmap import row_priority_weights, soft_bce_logits, target_weighted_mse


@dataclass
class TeacherLossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_floats(self) -> dict[str, float]:
        out = {"L_total": float(self.total.detach())}
        for k, v in self.terms.items():
            out[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
        return out


class TeacherLoss:
    """BCE-with-logits on ``Hhat_1s`` vs ``h_1s`` (heatmap at t+30).

    Occupied-cell pos weight matches Trial 3. Row weights apply only on the
    MSE path. Samples with ``amb_1s`` are dropped (unavailable 1s, or
    ``mask_ambiguous_pairs``). ``mean_target`` writes the averaged 1s map
    and clears that flag.
    """

    def __init__(self, cfg, device: str = "cpu"):
        self.map_w = float(cfg.get("loss.now_weight", 1.0))
        self.heatmap = str(cfg.get("loss.heatmap", "bce")).lower()
        self.bce_pos_threshold = float(cfg.get("loss.bce_pos_threshold", 0.1))
        self.bce_pos_weight = float(cfg.get("loss.bce_pos_weight", 2.0))
        self.floor = float(cfg.get("loss.threat_weight_floor", 0.05))
        self.scale = float(cfg.get("loss.threat_weight_scale", 4.0))
        self.gamma = float(cfg.get("loss.threat_weight_gamma", 2.0))
        self.recall_beta = float(cfg.get("loss.recall_weight_beta", 1.0))
        self.recall_hi = float(cfg.get("loss.recall_weight_hi", 0.5))
        self.row_list = list(cfg.get("loss.row_weights", [1.0, 1.0, 1.0, 1.0, 1.0]))
        self.device = device

    def __call__(self, out: dict[str, torch.Tensor], batch: dict, z_bg_pair: torch.Tensor | None = None) -> TeacherLossOutput:
        del z_bg_pair
        pred = out["Hhat_1s"]
        target = batch["h_1s"]
        sample_mask = None
        if "amb_1s" in batch:
            sample_mask = ~batch["amb_1s"].bool()
        row_w = row_priority_weights(self.row_list, pred.device, torch.float32)
        if self.heatmap == "mse":
            l_1s = target_weighted_mse(
                pred,
                target,
                sample_mask=sample_mask,
                row_weights=row_w,
                floor=self.floor,
                scale=self.scale,
                gamma=self.gamma,
                recall_beta=self.recall_beta,
                recall_hi=self.recall_hi,
            )
        else:
            l_1s = soft_bce_logits(
                out["Hhat_1s_logits"],
                target,
                sample_mask=sample_mask,
                pos_threshold=self.bce_pos_threshold,
                pos_weight=self.bce_pos_weight,
            )
        total = self.map_w * l_1s
        terms = {"L_1s": l_1s, "L_total": total}
        return TeacherLossOutput(total=total, terms=terms)
