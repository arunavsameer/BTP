"""Latent-only distillation. Heatmaps are never in the training loss.

V-JEPA / I-JEPA / SALT / V-JEPA 2.1 distill: L1 on predicted vs frozen-teacher F.
MGD: reconstruct teacher F from a spatially masked student context.
TinyMIM/SP: match pairwise spatial relations of F.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def spatial_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred.float() - target.float().detach()).abs().mean()


def spatial_cosine(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 − cosine of per-location channel vectors. Direction match, in [0, 2]."""
    p = pred.float().permute(0, 2, 3, 1).flatten(0, 2)
    t = target.float().detach().permute(0, 2, 3, 1).flatten(0, 2)
    p = F.normalize(p, dim=-1, eps=1e-6)
    t = F.normalize(t, dim=-1, eps=1e-6)
    return (1.0 - (p * t).sum(dim=-1)).mean()


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
        if teacher_out is not None:
            f_s = student_out["F"]
            f_t = teacher_out["F"]
            l_feat = spatial_l1(f_s, f_t)
            l_cos = spatial_cosine(f_s, f_t)
            l_rel = relation_loss(f_s, f_t)
            if "Z_t" in student_out and "Z_t" in teacher_out:
                l_z = spatial_l1(student_out["Z_t"], teacher_out["Z_t"])
            ctx = student_out.get("F_ctx")
            if predictor is not None and ctx is not None and self.mgd_w > 0:
                mask = random_spatial_mask(ctx, keep=self.mgd_keep)
                f_mgd = predictor(ctx, mask=mask)
                l_mgd = spatial_l1(f_mgd, f_t)
        total = (
            self.feat_w * l_feat
            + self.z_w * l_z
            + self.cos_w * l_cos
            + self.mgd_w * l_mgd
            + self.rel_w * l_rel
        )
        terms = {
            "L_feat": l_feat,
            "L_z": l_z,
            "L_cos": l_cos,
            "L_mgd": l_mgd,
            "L_rel": l_rel,
            "L_total": total,
        }
        return StudentLossOutput(total=total, terms=terms)
