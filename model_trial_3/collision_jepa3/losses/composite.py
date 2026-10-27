"""Composite teacher loss with independently logged terms."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..config import heads_mode, supervised_horizon_names
from .background_consistency import background_consistency_loss
from .beta_evidence import (
    beta_nll,
    evidential_mistake_penalty,
    prob_safe_hinge,
    prob_stop_hinge,
)
from .direction import direction_alignment_loss
from .heatmap import (
    appear_cell_weight,
    blob_bce,
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
        self.dir_thr = float(cfg.get("loss.direction_threat_threshold", 0.25))
        self.residual_mix = float(cfg.get("loss.delta_residual_mix", 0.5))
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
        self.beta_nll_w = float(cfg.get("loss.beta_nll_weight", 0.0))
        self.evidential_w = float(cfg.get("loss.evidential_weight", 0.0))
        self.prob_stop_w = float(cfg.get("loss.prob_stop_weight", 0.0))
        self.prob_safe_w = float(cfg.get("loss.prob_safe_weight", 0.0))
        self.prob_stop_target = float(cfg.get("loss.prob_stop_target", 0.75))
        self.prob_safe_cap = float(cfg.get("loss.prob_safe_cap", 0.15))
        self.row_list = list(cfg.get("adapter.row_weights", [1.0, 1.0, 1.0, 1.0, 1.0]))
        self.left = list(cfg.get("evaluation.left_cols", [0, 1]))
        self.center = list(cfg.get("evaluation.center_cols", [2]))
        self.right = list(cfg.get("evaluation.right_cols", [3, 4]))
        self.device = device

    def _row_w(self, ref: torch.Tensor) -> torch.Tensor:
        return row_priority_weights(self.row_list, ref.device, torch.float32)

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

    def _stop_hinge(self, pred, target, mask):
        if float(self.stop_hinge_w) == 0.0:
            return pred.new_zeros(())
        return stop_peak_hinge(
            pred,
            target,
            sample_mask=mask,
            stop_threshold=self.stop_hinge_threshold,
            margin=self.stop_hinge_margin,
            pred_target=self.stop_hinge_target,
        )

    def _safe_hinge(self, pred, target, mask):
        if float(self.safe_hinge_w) == 0.0:
            return pred.new_zeros(())
        return safe_peak_hinge(
            pred,
            target,
            sample_mask=mask,
            caution_threshold=self.safe_hinge_threshold,
            margin=self.safe_hinge_margin,
        )

    def _precision_logits(self, out: dict, ref: torch.Tensor) -> torch.Tensor:
        raw = out.get("Hhat_precision_logits")
        if raw is None:
            return ref.new_zeros(ref.shape)
        return raw

    def _beta_terms(self, pred, precision_logits, target, mask, cell_weight=None):
        if float(self.beta_nll_w) == 0.0:
            l_beta = pred.new_zeros(())
        else:
            l_beta = beta_nll(
                pred,
                precision_logits,
                target,
                sample_mask=mask,
                cell_weight=cell_weight,
            )
        if float(self.evidential_w) == 0.0:
            l_evi = pred.new_zeros(())
        else:
            l_evi = evidential_mistake_penalty(
                pred, precision_logits, target, sample_mask=mask
            )
        if float(self.prob_stop_w) == 0.0:
            l_pstop = pred.new_zeros(())
        else:
            l_pstop = prob_stop_hinge(
                pred,
                precision_logits,
                target,
                sample_mask=mask,
                gt_stop=self.stop_hinge_threshold,
                target_prob=self.prob_stop_target,
            )
        if float(self.prob_safe_w) == 0.0:
            l_psafe = pred.new_zeros(())
        else:
            l_psafe = prob_safe_hinge(
                pred,
                precision_logits,
                target,
                sample_mask=mask,
                caution=self.safe_hinge_threshold,
                cap_prob=self.prob_safe_cap,
                stop_threshold=self.stop_hinge_threshold,
            )
        return l_beta, l_evi, l_pstop, l_psafe

    def _map(
        self,
        pred,
        target,
        mask,
        row_w=None,
        extra=None,
        recall_beta: float = 1.0,
        cell_weight=None,
        logits=None,
    ):
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
            extra=extra,
            recall_beta=recall_beta,
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
        return blob_bce(
            pred,
            target,
            sample_mask=mask,
            occupied_threshold=self.blob_occupied_threshold,
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
            logits = out.get("Hhat_1s_logits")
            l_h = self._map(
                h_hat,
                h_tgt,
                mask=valid,
                row_w=row_w,
                recall_beta=self.recall_beta,
                cell_weight=self._cell_w(batch.get("h_now"), h_tgt),
                logits=logits,
            )
            l_blob = self._blob(h_hat, h_tgt, valid, logits=logits)
            l_stop = self._stop_hinge(h_hat, h_tgt, valid)
            l_safe = self._safe_hinge(h_hat, h_tgt, valid)
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
            l_close = self._close(
                out.get("Hhat_close"),
                batch.get("h_close"),
                batch.get("close_mask"),
                valid,
            )
            l_beta, l_evi, l_pstop, l_psafe = self._beta_terms(
                h_hat,
                self._precision_logits(out, h_hat),
                h_tgt,
                valid,
                cell_weight=self._cell_w(batch.get("h_now"), h_tgt),
            )
            terms["L_1s"] = l_h
            terms["L_future"] = l_h
            terms["L_blob"] = l_blob
            terms["L_blob_1s"] = l_blob
            terms["L_stop_hinge"] = l_stop
            terms["L_safe_hinge"] = l_safe
            terms["L_close"] = l_close
            terms["L_beta"] = l_beta
            terms["L_evidential"] = l_evi
            terms["L_prob_stop"] = l_pstop
            terms["L_prob_safe"] = l_psafe
            terms["L_FP"] = l_fp
            terms["L_FP_1s"] = l_fp
            terms["L_dir"] = l_dir
            terms["L_dir_1s"] = l_dir
            terms["L_bg"] = l_bg
            total = (
                l_h
                + self.blob_w * l_blob
                + self.stop_hinge_w * l_stop
                + self.safe_hinge_w * l_safe
                + self.close_w * l_close
                + self.beta_nll_w * l_beta
                + self.evidential_w * l_evi
                + self.prob_stop_w * l_pstop
                + self.prob_safe_w * l_psafe
                + self.fp_w * l_fp
                + self.dir_w * l_dir
                + self.bg_w * l_bg
            )
            terms["L_total"] = total
            return TeacherLossOutput(total=total, terms=terms)

        h_now_hat = out["Hhat_now"]
        row_w = self._row_w(h_now_hat)
        l_now = self._map(
            h_now_hat,
            batch["h_now"],
            mask=None,
            row_w=row_w,
            logits=out.get("Hhat_now_logits"),
        )
        terms["L_now"] = l_now

        future_terms = []
        blob_terms = []
        delta_terms = []
        fp_terms = []
        dir_terms = []
        for name in self.names:
            mask = batch[f"amb_{name}"]
            valid = ~mask if mask.dtype == torch.bool else (mask < 0.5)
            h_tgt = batch[f"h_{name}"]
            h_hat = out[f"Hhat_{name}"]
            l_h = self._map(
                h_hat,
                h_tgt,
                mask=valid,
                row_w=row_w,
                cell_weight=self._cell_w(batch.get("h_now"), h_tgt),
            )
            l_blob = self._blob(h_hat, h_tgt, valid)
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
            terms[f"L_blob_{name}"] = l_blob
            terms[f"L_delta_{name}"] = l_d
            terms[f"L_FP_{name}"] = l_fp
            terms[f"L_dir_{name}"] = l_dir
            future_terms.append(w * l_h)
            blob_terms.append(w * l_blob)
            delta_terms.append(w * l_d)
            fp_terms.append(w * l_fp)
            dir_terms.append(w * l_dir)

        l_future = sum(future_terms)
        l_blob = sum(blob_terms)
        l_delta = sum(delta_terms)
        l_fp = sum(fp_terms)
        l_dir = sum(dir_terms)
        terms["L_future"] = l_future
        terms["L_blob"] = l_blob
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
            + self.blob_w * l_blob
            + self.now_w * l_now
            + self.delta_w * l_delta
            + self.fp_w * l_fp
            + self.dir_w * l_dir
            + self.bg_w * l_bg
        )
        terms["L_total"] = total
        return TeacherLossOutput(total=total, terms=terms)
