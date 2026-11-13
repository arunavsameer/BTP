"""Student loss: L_feat distillation + Trial-3 G heatmap recipe.

Latent: channel-normalized L1 of student F vs cached parent F_dec only.
Heatmap: collision_jepa4 TeacherLoss (G): BCE with appear 8x / small-blob 2x,
blob BCE, STOP hinge, SAFE hinge, FP, direction. L_close is 0 because the
frozen Pair-CF decoder has no close head.

Component weights (feat_weight, heatmap_weight) are constants, default 1
and 1. G's internal heatmap term weights are also fixed.
"""

from __future__ import annotations

import torch

from collision_jepa4.losses import TeacherLoss


def blob_weights(f_t: torch.Tensor, h_1s: torch.Tensor | None = None) -> torch.Tensor:
    del h_1s
    b, _, h, w = f_t.shape
    return torch.ones(b, 1, h, w, device=f_t.device, dtype=torch.float32)


def weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float().detach()).abs().mean(dim=1, keepdim=True)
    return (weight.to(dtype=err.dtype) * err).mean()


def feat_loss(out, batch, mu, sigma) -> torch.Tensor:
    fs = out["F"]
    ft = batch["F_teacher"]
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    w = blob_weights(ft, batch.get("gt_1s", batch.get("h_1s")))
    return weighted_l1(fs_n, ft_n, w)


def heatmap_batch(batch: dict) -> dict:
    h_1s = batch.get("h_1s", batch.get("gt_1s"))
    out = {"h_1s": h_1s, "amb_1s": batch["amb_1s"]}
    if batch.get("h_now") is not None:
        out["h_now"] = batch["h_now"]
    return out


def student_loss(
    out,
    batch,
    *,
    mu,
    sigma,
    heatmap_loss: TeacherLoss,
    feat_weight: float,
    heatmap_weight: float,
):
    l_feat = feat_loss(out, batch, mu, sigma)
    hm = heatmap_loss(out, heatmap_batch(batch))
    l_hm = hm.total
    c_feat = float(feat_weight) * l_feat
    c_hm = float(heatmap_weight) * l_hm
    total = c_feat + c_hm
    den = total.detach().clamp_min(1e-8)
    terms = {
        "L_feat": float(l_feat.detach()),
        "L_heatmap": float(l_hm.detach()),
        "L_feat_contrib": float(c_feat.detach()),
        "L_heatmap_contrib": float(c_hm.detach()),
        "share_feat": float((c_feat.detach() / den).clamp(0, 1)),
        "share_heatmap": float((c_hm.detach() / den).clamp(0, 1)),
        "feat_weight": float(feat_weight),
        "heatmap_weight": float(heatmap_weight),
        "L_total": float(total.detach()),
    }
    for k, v in hm.terms.items():
        if k == "L_total":
            continue
        terms[k] = float(v.detach()) if torch.is_tensor(v) else float(v)
    return total, terms
