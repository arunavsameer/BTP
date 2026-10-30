"""Now-map BCE-with-logits. Future horizons stay in the batch and are not scored."""

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
    """Trial 3 ``L_now`` on ``Hhat_now``.

    BCE-with-logits uses the occupied-cell pos weight. Row weights are applied
    only on the MSE path, matching Trial 3 ``_map`` (uniform rows are a no-op
    for BCE).
    """

    def __init__(self, cfg, device: str = "cpu"):
        self.now_w = float(cfg.get("loss.now_weight", 1.0))
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
        pred = out["Hhat_now"]
        target = batch["h_now"]
        row_w = row_priority_weights(self.row_list, pred.device, torch.float32)
        if self.heatmap == "mse":
            l_now = target_weighted_mse(
                pred,
                target,
                sample_mask=None,
                row_weights=row_w,
                floor=self.floor,
                scale=self.scale,
                gamma=self.gamma,
                recall_beta=self.recall_beta,
                recall_hi=self.recall_hi,
            )
        else:
            l_now = soft_bce_logits(
                out["Hhat_now_logits"],
                target,
                sample_mask=None,
                pos_threshold=self.bce_pos_threshold,
                pos_weight=self.bce_pos_weight,
            )
        total = self.now_w * l_now
        terms = {"L_now": l_now, "L_total": total}
        return TeacherLossOutput(total=total, terms=terms)
