"""Trial 3 G teacher loss on the 1s-ahead trunk.

L_total = L_1s + 0.35 L_blob + 1.0 L_stop_hinge + 0.5 L_safe_hinge
        + 1.0 L_close + 0.10 L_FP + 0.05 L_dir
L_1s also applies appear 8× and small-blob 2× cell weights.
L_bg is logged as 0: trial 4 has no Z_t for background consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .direction import direction_alignment_loss
from .heatmap import (
    appear_cell_weight,
    blob_bce_logits,
    disappear_cell_weight,
    false_positive_penalty,
    row_priority_weights,
    safe_peak_hinge,
    small_blob_cell_weight,
    soft_bce,
    soft_bce_logits,
    stop_peak_hinge,
    target_weighted_mse,
)


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
    """G-recipe 1s loss. Occupied-cell pos weight matches Trial 3."""

    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.now_w = float(cfg.get("loss.now_weight", 1.0))
        self.fp_w = float(cfg.get("loss.false_positive_weight", 0.10))
        self.dir_w = float(cfg.get("loss.direction_weight", 0.05))
        self.bg_w = float(cfg.get("loss.background_consistency_weight", 0.02))
        self.safe_thr = float(cfg.get("loss.safe_cell_threshold", 0.05))
        self.fp_margin = float(cfg.get("loss.false_positive_margin", 0.10))
        self.dir_temp = float(cfg.get("loss.direction_temperature", 0.08))
        self.dir_thr = float(cfg.get("loss.direction_threat_threshold", 0.25))
        self.floor = float(cfg.get("loss.threat_weight_floor", 0.05))
        self.scale = float(cfg.get("loss.threat_weight_scale", 4.0))
        self.gamma = float(cfg.get("loss.threat_weight_gamma", 2.0))
        self.recall_beta = float(cfg.get("loss.recall_weight_beta", 1.0))
        self.recall_hi = float(cfg.get("loss.recall_weight_hi", 0.5))
        self.heatmap = str(cfg.get("loss.heatmap", "bce")).lower()
        self.bce_pos_threshold = float(cfg.get("loss.bce_pos_threshold", 0.1))
        self.bce_pos_weight = float(cfg.get("loss.bce_pos_weight", 2.0))
        self.appear_now_threshold = float(cfg.get("loss.bce_appear_now_threshold", 0.1))
        self.appear_future_threshold = float(cfg.get("loss.bce_appear_future_threshold", 0.5))
        self.appear_weight = float(cfg.get("loss.bce_appear_weight", 8.0))
        self.disappear_now_threshold = float(cfg.get("loss.bce_disappear_now_threshold", 0.5))
        self.disappear_future_threshold = float(cfg.get("loss.bce_disappear_future_threshold", 0.1))
        self.disappear_weight = float(cfg.get("loss.bce_disappear_weight", 1.0))
        self.close_w = float(cfg.get("loss.close_weight", 0.0))
        self.blob_w = float(cfg.get("loss.blob_bce_weight", 0.35))
        self.blob_occupied_threshold = float(cfg.get("loss.blob_occupied_threshold", 0.5))
        self.stop_hinge_w = float(cfg.get("loss.stop_hinge_weight", 0.0))
        self.stop_hinge_margin = float(cfg.get("loss.stop_hinge_margin", 0.05))
        self.stop_hinge_threshold = float(
            cfg.get("loss.stop_hinge_threshold", cfg.get("warning.stop_threshold", 0.60))
        )
        raw_target = cfg.get("loss.stop_hinge_target", None)
        self.stop_hinge_target = None if raw_target is None else float(raw_target)
        self.small_blob_w = float(cfg.get("loss.small_blob_weight", 1.0))
        self.small_blob_max_cells = int(cfg.get("loss.small_blob_max_cells", 4))
        self.small_blob_occupied_threshold = float(
            cfg.get("loss.small_blob_occupied_threshold", self.blob_occupied_threshold)
        )
        self.safe_hinge_w = float(cfg.get("loss.safe_hinge_weight", 0.0))
        self.safe_hinge_margin = float(cfg.get("loss.safe_hinge_margin", 0.0))
        self.safe_hinge_threshold = float(
            cfg.get("loss.safe_hinge_threshold", cfg.get("warning.caution_threshold", 0.25))
        )
        self.row_list = list(cfg.get("loss.row_weights", [1.0, 1.0, 1.0, 1.0, 1.0]))
        self.left = list(cfg.get("evaluation.left_cols", [0, 1]))
        self.center = list(cfg.get("evaluation.center_cols", [2]))
        self.right = list(cfg.get("evaluation.right_cols", [3, 4]))
        self.device = device

    def _row_w(self, ref: torch.Tensor) -> torch.Tensor:
        return row_priority_weights(self.row_list, ref.device, torch.float32)

    def _valid_mask(self, batch: dict) -> torch.Tensor | None:
        mask = batch.get("amb_1s")
        if mask is None:
            return None
        return ~mask if mask.dtype == torch.bool else (mask < 0.5)

    def _appear_w(self, h_now: torch.Tensor | None, h_tgt: torch.Tensor) -> torch.Tensor | None:
        if h_now is None or float(self.appear_weight) == 1.0:
            return None
        return appear_cell_weight(
            h_now,
            h_tgt,
            now_threshold=self.appear_now_threshold,
            future_threshold=self.appear_future_threshold,
            weight=self.appear_weight,
        )

    def _disappear_w(self, h_now: torch.Tensor | None, h_tgt: torch.Tensor) -> torch.Tensor | None:
        if h_now is None or float(self.disappear_weight) == 1.0:
            return None
        return disappear_cell_weight(
            h_now,
            h_tgt,
            now_threshold=self.disappear_now_threshold,
            future_threshold=self.disappear_future_threshold,
            weight=self.disappear_weight,
        )

    def _small_blob_w(self, h_tgt: torch.Tensor) -> torch.Tensor | None:
        if float(self.small_blob_w) == 1.0:
            return None
        return small_blob_cell_weight(
            h_tgt,
            stop_threshold=self.stop_hinge_threshold,
            occupied_threshold=self.small_blob_occupied_threshold,
            max_cells=self.small_blob_max_cells,
            weight=self.small_blob_w,
        )

    def _cell_w(self, h_now: torch.Tensor | None, h_tgt: torch.Tensor) -> torch.Tensor | None:
        parts = [self._appear_w(h_now, h_tgt), self._disappear_w(h_now, h_tgt), self._small_blob_w(h_tgt)]
        out = None
        for w in parts:
            if w is None:
                continue
            out = w if out is None else out * w
        return out

    def _map(self, pred, target, mask, row_w=None, cell_weight=None, logits=None):
        if self.heatmap != "mse":
            if logits is not None:
                return soft_bce_logits(
                    logits,
                    target,
                    sample_mask=mask,
                    pos_threshold=self.bce_pos_threshold,
                    pos_weight=self.bce_pos_weight,
                    cell_weight=cell_weight,
                )
            return soft_bce(
                pred,
                target,
                sample_mask=mask,
                pos_threshold=self.bce_pos_threshold,
                pos_weight=self.bce_pos_weight,
                cell_weight=cell_weight,
            )
        return target_weighted_mse(
            pred,
            target,
            sample_mask=mask,
            row_weights=row_w,
            floor=self.floor,
            scale=self.scale,
            gamma=self.gamma,
            recall_beta=self.recall_beta,
            recall_hi=self.recall_hi,
        )

    def _blob(self, pred, target, mask, logits=None):
        if self.heatmap == "mse" or float(self.blob_w) == 0.0:
            return pred.new_zeros(())
        if logits is not None:
            return blob_bce_logits(
                logits,
                target,
                sample_mask=mask,
                occupied_threshold=self.blob_occupied_threshold,
            )
        from .heatmap import blob_bce

        return blob_bce(
            pred,
            target,
            sample_mask=mask,
            occupied_threshold=self.blob_occupied_threshold,
        )

    def _close(self, pred, target, occ, mask):
        if float(self.close_w) == 0.0 or pred is None or target is None:
            if pred is not None:
                return pred.new_zeros(())
            if target is not None:
                return target.new_zeros(())
            return torch.zeros(())
        return soft_bce(
            pred,
            target,
            sample_mask=mask,
            pos_threshold=0.5,
            pos_weight=2.0,
            cell_weight=occ,
        )

    def __call__(
        self,
        out: dict[str, torch.Tensor],
        batch: dict,
        z_bg_pair: torch.Tensor | None = None,
    ) -> TeacherLossOutput:
        del z_bg_pair
        h_hat = out["Hhat_1s"]
        h_tgt = batch["h_1s"]
        logits = out.get("Hhat_1s_logits")
        valid = self._valid_mask(batch)
        row_w = self._row_w(h_hat)
        l_h = self._map(
            h_hat,
            h_tgt,
            mask=valid,
            row_w=row_w,
            cell_weight=self._cell_w(batch.get("h_now"), h_tgt),
            logits=logits,
        )
        l_blob = self._blob(h_hat, h_tgt, valid, logits=logits)
        if float(self.stop_hinge_w) == 0.0:
            l_stop = h_hat.new_zeros(())
        else:
            l_stop = stop_peak_hinge(
                h_hat,
                h_tgt,
                sample_mask=valid,
                stop_threshold=self.stop_hinge_threshold,
                margin=self.stop_hinge_margin,
                pred_target=self.stop_hinge_target,
            )
        if float(self.safe_hinge_w) == 0.0:
            l_safe = h_hat.new_zeros(())
        else:
            l_safe = safe_peak_hinge(
                h_hat,
                h_tgt,
                sample_mask=valid,
                caution_threshold=self.safe_hinge_threshold,
                margin=self.safe_hinge_margin,
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
        l_close = self._close(
            out.get("Hhat_close"),
            batch.get("h_close"),
            batch.get("close_mask"),
            valid,
        )
        l_bg = h_hat.new_zeros(())
        total = (
            self.now_w * l_h
            + self.blob_w * l_blob
            + self.stop_hinge_w * l_stop
            + self.safe_hinge_w * l_safe
            + self.close_w * l_close
            + self.fp_w * l_fp
            + self.dir_w * l_dir
            + self.bg_w * l_bg
        )
        terms = {
            "L_1s": l_h,
            "L_blob": l_blob,
            "L_stop_hinge": l_stop,
            "L_safe_hinge": l_safe,
            "L_close": l_close,
            "L_FP": l_fp,
            "L_dir": l_dir,
            "L_bg": l_bg,
            "L_total": total,
        }
        return TeacherLossOutput(total=total, terms=terms)
