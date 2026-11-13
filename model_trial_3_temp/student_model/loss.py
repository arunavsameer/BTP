"""Latent-only distillation with blob/appear-weighted F matching.

Uniform L1 lets large jaywalker maps drown 1–2 cell appear blobs.
Weights follow teacher-F energy plus GT appear/STOP cells (weights only; H is not a target).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def spatial_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float().detach()).abs().mean()


def blob_weights(
    f_t: torch.Tensor,
    h_now: torch.Tensor | None = None,
    h_1s: torch.Tensor | None = None,
    *,
    energy_alpha: float = 4.0,
    appear_boost: float = 4.0,
    stop_boost: float = 2.0,
) -> torch.Tensor:
    """Per-location weights [B,1,H,W] on the 16×16 F grid. Mean-normalized to 1."""
    energy = f_t.detach().float().pow(2).mean(dim=1, keepdim=True)
    energy = energy / energy.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    w = 1.0 + float(energy_alpha) * energy
    if h_1s is not None:
        occ = F.interpolate(h_1s.detach().float().unsqueeze(1), size=energy.shape[-2:], mode="nearest")
        w = w + float(stop_boost) * (occ >= 0.60).to(w.dtype)
        if h_now is not None:
            now = F.interpolate(h_now.detach().float().unsqueeze(1), size=energy.shape[-2:], mode="nearest")
            appear = (occ >= 0.50) & (now < 0.25)
            w = w + float(appear_boost) * appear.to(w.dtype)
    return w / w.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float().detach()).abs().mean(dim=1, keepdim=True)
    return (weight.to(dtype=err.dtype) * err).mean()


def spatial_cosine(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    """1 − cosine of per-location channel vectors."""
    p = pred.float().permute(0, 2, 3, 1)
    t = target.float().detach().permute(0, 2, 3, 1)
    p = F.normalize(p, dim=-1, eps=1e-6)
    t = F.normalize(t, dim=-1, eps=1e-6)
    loc = 1.0 - (p * t).sum(dim=-1)
    if weight is None:
        return loc.mean()
    w = weight.squeeze(1).to(dtype=loc.dtype)
    return (w * loc).sum() / w.sum().clamp(min=1e-6)


def relation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Similarity-preserving: student HW×HW gram vs teacher (Tian SP / TinyMIM-style)."""
    p = F.normalize(pred.float().flatten(2), dim=1)
    t = F.normalize(target.float().detach().flatten(2), dim=1)
    gp = torch.bmm(p.transpose(1, 2), p)
    gt = torch.bmm(t.transpose(1, 2), t)
    return F.mse_loss(gp, gt)


def random_spatial_mask(fine: torch.Tensor, keep: float = 0.5) -> torch.Tensor:
    b, _, h, w = fine.shape
    keep_p = float(keep)
    mask = (torch.rand(b, 1, h, w, device=fine.device, dtype=fine.dtype) < keep_p).to(fine.dtype)
    return mask


@dataclass
class StudentLossOutput:
    total: torch.Tensor
    terms: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_floats(self) -> dict[str, float]:
        out = {"L_total": float(self.total.detach())}
        for k, v in self.terms.items():
            out[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
        return out


class StudentLoss:
    def __init__(self, cfg, device: str = "cpu"):
        self.cfg = cfg
        self.feat_w = float(cfg.get("student.feat_weight", 1.0))
        self.z_w = float(cfg.get("student.z_weight", 0.3))
        self.cos_w = float(cfg.get("student.cos_weight", 0.2))
        self.mgd_w = float(cfg.get("student.mgd_weight", 0.5))
        self.rel_w = float(cfg.get("student.rel_weight", 0.2))
        self.mgd_keep = float(cfg.get("student.mgd_keep", 0.5))
        self.energy_alpha = float(cfg.get("student.energy_alpha", 0.0))
        self.appear_boost = float(cfg.get("student.appear_boost", 0.0))
        self.stop_boost = float(cfg.get("student.stop_boost", 0.0))
        self.h_w = float(cfg.get("student.h_weight", 0.0))
        self.logit_w = float(cfg.get("student.logit_weight", 0.0))
        self.peak_stop_w = float(cfg.get("student.peak_stop_weight", 0.0))
        self.peak_safe_w = float(cfg.get("student.peak_safe_weight", 0.0))
        self.stop_thr = float(cfg.get("warning.stop_threshold", 0.60))
        self.caution_thr = float(cfg.get("warning.caution_threshold", 0.25))
        self.stop_target = float(cfg.get("student.stop_hinge_target", 0.65))

    def __call__(
        self,
        student_out: dict[str, torch.Tensor],
        batch: dict,
        teacher_out: dict[str, torch.Tensor] | None,
        *,
        epoch: int = 0,
        predictor=None,
    ) -> StudentLossOutput:
        zero = student_out["F"].new_zeros(())
        l_feat = zero
        l_z = zero
        l_cos = zero
        l_mgd = zero
        l_rel = zero
        l_h = zero
        l_logit = zero
        l_stop = zero
        l_safe = zero
        if teacher_out is not None:
            f_s = student_out["F"]
            f_t = teacher_out["F"]
            h_now = batch.get("h_now")
            h_1s = batch.get("h_1s")
            w = blob_weights(
                f_t,
                h_now,
                h_1s,
                energy_alpha=self.energy_alpha,
                appear_boost=self.appear_boost,
                stop_boost=self.stop_boost,
            )
            l_feat = weighted_l1(f_s, f_t, w)
            l_cos = spatial_cosine(f_s, f_t, w)
            l_rel = relation_loss(f_s, f_t)
            if "Z_t" in student_out and "Z_t" in teacher_out:
                z_w = None
                if h_1s is not None:
                    z_w = 1.0 + self.stop_boost * (h_1s.detach().float().unsqueeze(1) >= 0.60).to(f_s.dtype)
                    z_w = z_w / z_w.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)
                    l_z = weighted_l1(student_out["Z_t"], teacher_out["Z_t"], z_w)
                else:
                    l_z = spatial_l1(student_out["Z_t"], teacher_out["Z_t"])
            ctx = student_out.get("F_ctx")
            if predictor is not None and ctx is not None and self.mgd_w > 0:
                mask = random_spatial_mask(ctx, keep=self.mgd_keep)
                f_mgd = predictor(ctx, mask=mask)
                l_mgd = weighted_l1(f_mgd, f_t, w)
            h_s = student_out.get("Hhat_1s")
            h_t = teacher_out.get("Hhat_1s")
            if h_s is not None and h_t is not None and (
                self.h_w > 0 or self.logit_w > 0 or self.peak_stop_w > 0 or self.peak_safe_w > 0
            ):
                h_td = h_t.float().detach()
                h_sf = h_s.float()
                # Unweighted: empty cells are the majority. Mean-normalizing STOP
                # boosts (lat3) starved SAFE and let a global peak shortcut win.
                l_h = (h_sf - h_td).abs().mean()
                lg_s = student_out.get("Hhat_1s_logits")
                lg_t = teacher_out.get("Hhat_1s_logits")
                if lg_s is not None and lg_t is not None:
                    l_logit = (lg_s.float() - lg_t.float().detach()).abs().mean()
                stop_cells = h_td >= self.stop_thr
                safe_cells = h_td < self.caution_thr
                if bool(stop_cells.any()):
                    l_stop = (self.stop_target - h_sf[stop_cells]).clamp(min=0).mean()
                if bool(safe_cells.any()):
                    l_safe = (h_sf[safe_cells] - self.caution_thr).clamp(min=0).mean()
        total = (
            self.feat_w * l_feat
            + self.z_w * l_z
            + self.cos_w * l_cos
            + self.mgd_w * l_mgd
            + self.rel_w * l_rel
            + self.h_w * l_h
            + self.logit_w * l_logit
            + self.peak_stop_w * l_stop
            + self.peak_safe_w * l_safe
        )
        terms = {
            "L_feat": l_feat,
            "L_z": l_z,
            "L_cos": l_cos,
            "L_mgd": l_mgd,
            "L_rel": l_rel,
            "L_h": l_h,
            "L_logit": l_logit,
            "L_stop": l_stop,
            "L_safe": l_safe,
            "L_total": total,
        }
        return StudentLossOutput(total=total, terms=terms)
