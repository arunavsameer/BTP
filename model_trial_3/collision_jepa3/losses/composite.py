"""Composite teacher loss with independently logged terms."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import heads_mode, supervised_horizon_names
from .background_consistency import background_consistency_loss
from .direction import direction_alignment_loss
from .heatmap import false_positive_penalty, row_priority_weights, target_weighted_mse
from .residual import delta_loss


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
    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.direct_h1s = heads_mode(cfg) == "direct_h1s"
        self.names = supervised_horizon_names(cfg)
        hw = cfg.get("loss.horizon_weights")
        self.horizon_w = {n: float(hw[n]) for n in self.names}
        self.now_w = float(cfg.get("loss.now_weight", 0.25))
        self.delta_w = float(cfg.get("loss.delta_weight", 0.20))
        self.fp_w = float(cfg.get("loss.false_positive_weight", 0.10))
        self.dir_w = float(cfg.get("loss.direction_weight", 0.05))
        self.bg_w = float(cfg.get("loss.background_consistency_weight", 0.02))
        self.floor = float(cfg.get("loss.threat_weight_floor", 0.05))
        self.scale = float(cfg.get("loss.threat_weight_scale", 4.0))
        self.gamma = float(cfg.get("loss.threat_weight_gamma", 2.0))
        self.safe_thr = float(cfg.get("loss.safe_cell_threshold", 0.05))
        self.fp_margin = float(cfg.get("loss.false_positive_margin", 0.10))
        self.dir_temp = float(cfg.get("loss.direction_temperature", 0.08))
        self.dir_thr = float(cfg.get("loss.direction_threat_threshold", 0.42))
        self.residual_mix = float(cfg.get("loss.delta_residual_mix", 0.5))
        self.recall_beta = float(cfg.get("loss.recall_weight_beta", 1.0))
        self.recall_hi = float(cfg.get("loss.recall_weight_hi", 0.5))
        self.row_list = list(cfg.get("adapter.row_weights", [0.8, 0.9, 1.0, 1.1, 1.2]))
        self.left = list(cfg.get("evaluation.left_cols", [0, 1]))
        self.center = list(cfg.get("evaluation.center_cols", [2]))
        self.right = list(cfg.get("evaluation.right_cols", [3, 4]))
        self.device = device

    def _row_w(self, ref: torch.Tensor) -> torch.Tensor:
        return row_priority_weights(self.row_list, ref.device, torch.float32)

    def _map(self, pred, target, mask, row_w, extra=None, recall_beta: float = 1.0):
        return target_weighted_mse(
            pred,
            target,
            sample_mask=mask,
            row_weights=row_w,
            floor=self.floor,
            scale=self.scale,
            gamma=self.gamma,
            extra=extra,
            recall_beta=recall_beta,
            recall_hi=self.recall_hi,
        )

    def __call__(
        self,
        out: dict[str, torch.Tensor],
        batch: dict,
        z_bg_pair: torch.Tensor | None = None,
    ) -> TeacherLossOutput:
        terms: dict[str, torch.Tensor] = {}
        if self.direct_h1s:
            h_hat = out["Hhat_1s"]
            row_w = self._row_w(h_hat)
            mask = batch["amb_1s"]
            valid = ~mask if mask.dtype == torch.bool else (mask < 0.5)
            h_tgt = batch["h_1s"]
            l_h = self._map(h_hat, h_tgt, mask=valid, row_w=row_w, recall_beta=self.recall_beta)
            l_fp = false_positive_penalty(
                h_hat, h_tgt, sample_mask=valid, safe_threshold=self.safe_thr, margin=self.fp_margin
            )
            l_dir = direction_alignment_loss(
                h_hat,
                h_tgt,
                row_weights=row_w,
                left_cols=self.left,
                center_cols=self.center,
                right_cols=self.right,
                sample_mask=valid,
                temperature=self.dir_temp,
                threat_threshold=self.dir_thr,
            )
            if z_bg_pair is not None:
                l_bg = background_consistency_loss(out["Z_t"], z_bg_pair)
            else:
                l_bg = h_hat.new_zeros(())
            terms["L_1s"] = l_h
            terms["L_future"] = l_h
            terms["L_FP"] = l_fp
            terms["L_FP_1s"] = l_fp
            terms["L_dir"] = l_dir
            terms["L_dir_1s"] = l_dir
            terms["L_bg"] = l_bg
            total = l_h + self.fp_w * l_fp + self.dir_w * l_dir + self.bg_w * l_bg
            terms["L_total"] = total
            return TeacherLossOutput(total=total, terms=terms)

        h_now_hat = out["Hhat_now"]
        row_w = self._row_w(h_now_hat)
        l_now = self._map(h_now_hat, batch["h_now"], mask=None, row_w=row_w)
        terms["L_now"] = l_now

        future_terms = []
        delta_terms = []
        fp_terms = []
        dir_terms = []
        for name in self.names:
            mask = batch[f"amb_{name}"]
            valid = ~mask if mask.dtype == torch.bool else (mask < 0.5)
            h_tgt = batch[f"h_{name}"]
            h_hat = out[f"Hhat_{name}"]
            l_h = self._map(h_hat, h_tgt, mask=valid, row_w=row_w)
            l_d = delta_loss(
                out[f"Delta_{name}"],
                h_tgt,
                h_now_hat,
                sample_mask=valid,
                row_weights=row_w,
                floor=self.floor,
                scale=self.scale,
                gamma=self.gamma,
                residual_mix=self.residual_mix,
            )
            l_fp = false_positive_penalty(
                h_hat, h_tgt, sample_mask=valid, safe_threshold=self.safe_thr, margin=self.fp_margin
            )
            l_dir = direction_alignment_loss(
                h_hat,
                h_tgt,
                row_weights=row_w,
                left_cols=self.left,
                center_cols=self.center,
                right_cols=self.right,
                sample_mask=valid,
                temperature=self.dir_temp,
                threat_threshold=self.dir_thr,
            )
            w = self.horizon_w[name]
            terms[f"L_{name}"] = l_h
            terms[f"L_delta_{name}"] = l_d
            terms[f"L_FP_{name}"] = l_fp
            terms[f"L_dir_{name}"] = l_dir
            future_terms.append(w * l_h)
            delta_terms.append(w * l_d)
            fp_terms.append(w * l_fp)
            dir_terms.append(w * l_dir)

        l_future = sum(future_terms)
        l_delta = sum(delta_terms)
        l_fp = sum(fp_terms)
        l_dir = sum(dir_terms)
        terms["L_future"] = l_future
        terms["L_delta"] = l_delta
        terms["L_FP"] = l_fp
        terms["L_dir"] = l_dir

        if z_bg_pair is not None:
            l_bg = background_consistency_loss(out["Z_t"], z_bg_pair)
        else:
            l_bg = h_now_hat.new_zeros(())
        terms["L_bg"] = l_bg

        total = (
            l_future
            + self.now_w * l_now
            + self.delta_w * l_delta
            + self.fp_w * l_fp
            + self.dir_w * l_dir
            + self.bg_w * l_bg
        )
        terms["L_total"] = total
        return TeacherLossOutput(total=total, terms=terms)
