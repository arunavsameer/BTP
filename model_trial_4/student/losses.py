"""Latent-only distillation against cached parent F_dec."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def blob_weights(f_t: torch.Tensor, h_1s: torch.Tensor | None = None) -> torch.Tensor:
    energy = f_t.detach().float().pow(2).mean(dim=1, keepdim=True)
    energy = energy / energy.mean(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
    w = 1.0 + 0.0 * energy
    if h_1s is not None:
        occ = F.interpolate(h_1s.detach().float().unsqueeze(1), size=energy.shape[-2:], mode="nearest")
        w = w + 0.0 * (occ >= 0.60).to(w.dtype)
    return w / w.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-6)


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float().detach()).abs().mean(dim=1, keepdim=True)
    return (weight.to(dtype=err.dtype) * err).mean()


def spatial_cosine(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
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
    p = F.normalize(pred.float().flatten(2), dim=1)
    t = F.normalize(target.float().detach().flatten(2), dim=1)
    gp = torch.bmm(p.transpose(1, 2), p)
    gt = torch.bmm(t.transpose(1, 2), t)
    return F.mse_loss(gp, gt)


def random_spatial_mask(fine: torch.Tensor, keep: float = 0.5) -> torch.Tensor:
    b, _, h, w = fine.shape
    return (torch.rand(b, 1, h, w, device=fine.device, dtype=fine.dtype) < float(keep)).to(fine.dtype)


def latent_loss(out, batch, *, feat_w, cos_w, mgd_w, rel_w, mu, sigma, predictor, mgd_keep=0.5):
    fs = out["F"]
    ft = batch["F_teacher"]
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    w = blob_weights(ft, batch.get("gt_1s"))
    l_feat = weighted_l1(fs_n, ft_n, w)
    l_cos = spatial_cosine(fs, ft, w)
    l_rel = relation_loss(fs, ft)
    l_mgd = fs.new_zeros(())
    if mgd_w > 0 and predictor is not None:
        mask = random_spatial_mask(out["F_ctx"], keep=mgd_keep)
        f_mgd = predictor(out["F_ctx"], mask=mask)
        l_mgd = weighted_l1((f_mgd - mu) / (sigma + 1e-6), ft_n, w)
    total = feat_w * l_feat + cos_w * l_cos + mgd_w * l_mgd + rel_w * l_rel
    return total, {
        "L_feat": float(l_feat.detach()),
        "L_cos": float(l_cos.detach()),
        "L_mgd": float(l_mgd.detach()),
        "L_rel": float(l_rel.detach()),
        "L_total": float(total.detach()),
    }
