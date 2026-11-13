"""Task losses for overnight Pair-CF students. Heatmaps are 5×5 occupancy."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from student_model.loss import blob_weights, relation_loss, spatial_cosine, weighted_l1

STOP, THREAT, SAFE = 2, 1, 0


def peak_class_t(gt: torch.Tensor) -> torch.Tensor:
    p = gt.flatten(1).max(dim=1).values
    y = torch.zeros(gt.size(0), dtype=torch.long, device=gt.device)
    y = torch.where(p >= 0.25, torch.full_like(y, THREAT), y)
    y = torch.where(p >= 0.60, torch.full_like(y, STOP), y)
    return y


def _sample_reduce(per: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
    return (per * sw).sum() / sw.sum().clamp_min(1e-6)


def _per_sample_wl1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = (pred.float() - target.float().detach()).abs().mean(dim=1, keepdim=True)
    w = weight.to(dtype=err.dtype)
    return (w * err).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1e-6)


def _per_sample_cosine(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.float().permute(0, 2, 3, 1), dim=-1, eps=1e-6)
    t = F.normalize(target.float().detach().permute(0, 2, 3, 1), dim=-1, eps=1e-6)
    loc = 1.0 - (p * t).sum(dim=-1)
    w = weight.squeeze(1).to(dtype=loc.dtype)
    return (w * loc).flatten(1).sum(1) / w.flatten(1).sum(1).clamp_min(1e-6)


def _per_sample_rel(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.float().flatten(2), dim=1)
    t = F.normalize(target.float().detach().flatten(2), dim=1)
    gp = torch.bmm(p.transpose(1, 2), p)
    gt = torch.bmm(t.transpose(1, 2), t)
    return (gp - gt).pow(2).mean(dim=(1, 2))


def latent_sample_weights(gt: torch.Tensor, pre_fork: torch.Tensor) -> torch.Tensor:
    y = peak_class_t(gt)
    stop = y == STOP
    pref = pre_fork.to(dtype=torch.bool)
    w = torch.ones(gt.size(0), device=gt.device, dtype=gt.dtype)
    w = w + 3.0 * stop.to(w.dtype)
    w = w + 2.0 * (stop & pref).to(w.dtype)
    return w


def parent_correct_stop_kd(h: torch.Tensor, ht: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    yg, yp = peak_class_t(gt), peak_class_t(ht)
    m = (yg == STOP) & (yp == STOP)
    if not bool(m.any()):
        return h.new_zeros(())
    return ((h - ht).abs()[m]).mean()


def safe_peak_hinge(h: torch.Tensor, gt: torch.Tensor, cut: float = 0.25) -> torch.Tensor:
    y = peak_class_t(gt)
    p = h.flatten(1).max(1).values
    m = y == SAFE
    if not bool(m.any()):
        return h.new_zeros(())
    return F.relu(p[m] - cut).mean()


def collapse_recovery_losses(out, batch, mu, sigma, predictor, decoder, dec_init):
    """M1 latent weights + STOP spatial boost + parent-correct STOP KD + SAFE brake."""
    fs, ft = out["F"], batch["F_teacher"]
    gt = batch["gt_1s"].float()
    ht = batch["H_teacher"].float()
    h = out["Hhat_1s"].float()
    pref = batch["pre_fork"]
    w = blob_weights(ft, None, gt, energy_alpha=0.0, appear_boost=0.0, stop_boost=2.0)
    sw = latent_sample_weights(gt, pref)
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    lf = _sample_reduce(_per_sample_wl1(fs_n, ft_n, w), sw)
    lc = _sample_reduce(_per_sample_cosine(fs, ft, w), sw)
    lr = _sample_reduce(_per_sample_rel(fs, ft), sw)
    mask = (torch.rand(fs.size(0), 1, fs.size(2), fs.size(3), device=fs.device) < 0.5).to(fs.dtype)
    f_mgd = predictor(out["F_ctx"], mask=mask)
    if f_mgd.shape != fs_n.shape:
        f_mgd = F.interpolate(f_mgd, size=fs_n.shape[-2:], mode="bilinear", align_corners=False)
    lm = _sample_reduce(_per_sample_wl1((f_mgd - mu) / (sigma + 1e-6), ft_n, w), sw)
    lh = occupied_heatmap(h, gt)
    lfl = stop_floor(h, gt, 0.75)
    lkd = parent_correct_stop_kd(h, ht, gt)
    lsf = safe_peak_hinge(h, gt, 0.25)
    l_reg = decoder_anchor(decoder, dec_init)
    parts = {
        "L_feat": lf,
        "L_cos": lc,
        "L_rel": lr,
        "L_mgd": lm,
        "L_h": lh,
        "L_floor": lfl,
        "L_kd": lkd,
        "L_safe": lsf,
        "L_dec_reg": l_reg,
    }
    total = (
        1.0 * lf
        + 0.2 * lc
        + 0.5 * lm
        + 0.2 * lr
        + 0.8 * lh
        + 1.8 * lfl
        + 1.0 * lkd
        + 0.4 * lsf
        + 0.01 * l_reg
    )
    parts["L_total_base"] = total
    return parts, total, peak_class_t(gt)


def latent_terms(out, batch, mu, sigma, predictor):
    fs, ft = out["F"], batch["F_teacher"]
    w = blob_weights(ft, None, batch["gt_1s"], energy_alpha=0.0, appear_boost=0.0, stop_boost=0.0)
    fs_n = (fs - mu) / (sigma + 1e-6)
    ft_n = (ft - mu) / (sigma + 1e-6)
    l_feat = weighted_l1(fs_n, ft_n, w)
    l_cos = spatial_cosine(fs, ft, w)
    l_rel = relation_loss(fs, ft)
    mask = (torch.rand(fs.size(0), 1, fs.size(2), fs.size(3), device=fs.device) < 0.5).to(fs.dtype)
    f_mgd = predictor(out["F_ctx"], mask=mask)
    if f_mgd.shape != fs_n.shape:
        f_mgd = F.interpolate(f_mgd, size=fs_n.shape[-2:], mode="bilinear", align_corners=False)
    l_mgd = weighted_l1((f_mgd - mu) / (sigma + 1e-6), ft_n, w)
    return l_feat, l_cos, l_rel, l_mgd


def occupied_heatmap(h, gt, empty_w: float = 0.12, occ_w: float = 4.0):
    occ = (gt >= 0.25).to(gt.dtype)
    w = empty_w * (1.0 - occ) + occ_w * occ * (0.5 + gt.clamp(0, 1))
    return (F.smooth_l1_loss(h, gt, reduction="none", beta=0.05) * w).mean()


def stop_floor(h, gt, margin: float = 0.70):
    y = peak_class_t(gt)
    p = h.flatten(1).max(1).values
    m = y == STOP
    if not bool(m.any()):
        return h.new_zeros(())
    return F.relu(margin - p[m]).mean()


def selective_kd(h, ht, gt):
    yg, yp = peak_class_t(gt), peak_class_t(ht)
    agree = (yg == yp).to(h.dtype)
    contradict_stop = ((yp == STOP) & (yg != STOP)).to(h.dtype)
    w = agree * (1.0 - contradict_stop)
    den = w.sum().clamp_min(1.0)
    return ((h - ht).abs() * w.view(-1, 1, 1)).sum() / (den * 25.0)


def ordinal_peak(h, gt):
    y = peak_class_t(gt)
    p = h.flatten(1).max(1).values
    loss = h.new_zeros(())
    n = 0
    m = y == SAFE
    if bool(m.any()):
        loss = loss + F.relu(p[m] - 0.20).mean()
        n += 1
    m = y == THREAT
    if bool(m.any()):
        loss = loss + (F.relu(0.30 - p[m]) + F.relu(p[m] - 0.55)).mean()
        n += 1
    m = y == STOP
    if bool(m.any()):
        loss = loss + F.relu(0.70 - p[m]).mean()
        n += 1
    return loss / max(n, 1)


def hard_occupied(h, gt):
    occ = gt >= 0.25
    miss = F.relu(gt - h)
    if not bool(occ.any()):
        return occupied_heatmap(h, gt)
    vals = miss[occ]
    thr = torch.quantile(vals.detach(), 0.70)
    hard = occ & (miss >= thr)
    w = 0.12 + 4.0 * occ.to(gt.dtype) * (0.5 + gt) + 3.0 * hard.to(gt.dtype)
    return (F.smooth_l1_loss(h, gt, reduction="none", beta=0.05) * w).mean()


def recovery_weight(h, ht, gt):
    yg, yp, ys = peak_class_t(gt), peak_class_t(ht), peak_class_t(h.detach())
    w = torch.ones(gt.size(0), device=gt.device, dtype=gt.dtype)
    rec = (yg == STOP) & (yp == STOP) & (ys != STOP)
    parent_miss = (yg == STOP) & (yp != STOP)
    parent_false = (yg != STOP) & (yp == STOP)
    w = torch.where(rec, w * 5.0, w)
    w = torch.where(parent_miss, w * 1.5, w)
    w = torch.where(parent_false, w * 1.2, w)
    return w, rec


def ranking_stop_vs_safe(peaks, y, margin=0.20):
    stop, safe = y == STOP, y == SAFE
    if not bool(stop.any()) or not bool(safe.any()):
        return peaks.new_zeros(())
    return F.relu(margin - (peaks[stop].unsqueeze(1) - peaks[safe].unsqueeze(0))).mean()


def focal_ce(logits, y, alpha, gamma=2.0):
    logp = F.log_softmax(logits, dim=1)
    p = logp.exp()
    pt = p.gather(1, y.view(-1, 1)).squeeze(1)
    logpt = logp.gather(1, y.view(-1, 1)).squeeze(1)
    w = alpha.gather(0, y)
    return (-w * (1.0 - pt).clamp(0, 1).pow(gamma) * logpt).mean()


def decoder_anchor(decoder: torch.nn.Module, init: dict[str, torch.Tensor]) -> torch.Tensor:
    acc = None
    n = 0
    for name, p in decoder.named_parameters():
        t0 = init[name]
        d = (p - t0).pow(2).mean()
        acc = d if acc is None else acc + d
        n += 1
    return acc / max(n, 1)


def compute_losses(mid: int, out, batch, mu, sigma, predictor, decoder, dec_init, class_w):
    lf, lc, lr, lm = latent_terms(out, batch, mu, sigma, predictor)
    h = out["Hhat_1s"].float()
    gt = batch["gt_1s"].float()
    ht = batch["H_teacher"].float()
    y = peak_class_t(gt)
    l_reg = decoder_anchor(decoder, dec_init)
    parts = {"L_feat": lf, "L_cos": lc, "L_rel": lr, "L_mgd": lm, "L_dec_reg": l_reg}

    if mid == 1:
        lh = occupied_heatmap(h, gt)
        lfl = stop_floor(h, gt, 0.70)
        lkd = selective_kd(h, ht, gt)
        parts.update(L_h=lh, L_floor=lfl, L_kd=lkd)
        total = 1.0 * lf + 0.2 * lc + 0.5 * lm + 0.2 * lr + 0.8 * lh + 0.6 * lfl + 0.4 * lkd + 0.01 * l_reg
    elif mid == 2:
        lo = ordinal_peak(h, gt)
        lh = occupied_heatmap(h, gt, empty_w=0.08, occ_w=2.0)
        parts.update(L_ord=lo, L_h=lh)
        total = 0.6 * lf + 0.2 * lc + 0.3 * lm + 0.15 * lr + 1.2 * lo + 0.4 * lh + 0.01 * l_reg
    elif mid == 3:
        lh = hard_occupied(h, gt)
        lfl = stop_floor(h, gt, 0.70)
        empty = ((gt < 0.25).to(gt.dtype) * h.pow(2)).mean()
        parts.update(L_h=lh, L_floor=lfl, L_empty=empty)
        total = 0.8 * lf + 0.2 * lc + 0.4 * lm + 0.15 * lr + 1.0 * lh + 0.7 * lfl + 0.25 * empty + 0.01 * l_reg
    elif mid == 4:
        w, _ = recovery_weight(h, ht, gt)
        lh = occupied_heatmap(h, gt)
        lfl = (stop_floor(h, gt, 0.70) * (1.0 + 0.5 * (w.mean() - 1.0).clamp(min=0)))
        lkd = selective_kd(h, ht, gt)
        sl1 = F.smooth_l1_loss(h, gt, reduction="none", beta=0.05).mean(dim=(1, 2))
        lh_w = (sl1 * w).mean()
        parts.update(L_h=lh, L_h_rec=lh_w, L_floor=lfl, L_kd=lkd)
        total = 0.8 * lf + 0.2 * lc + 0.4 * lm + 0.15 * lr + 0.6 * lh + 0.8 * lh_w + 0.7 * lfl + 0.5 * lkd + 0.01 * l_reg
    elif mid == 5:
        lh = occupied_heatmap(h, gt)
        lfl = stop_floor(h, gt, 0.70)
        empty = ((gt < 0.25).to(gt.dtype) * h.pow(2)).mean()
        parts.update(L_h=lh, L_floor=lfl, L_empty=empty)
        total = 0.9 * lf + 0.2 * lc + 0.4 * lm + 0.15 * lr + 0.9 * lh + 0.6 * lfl + 0.2 * empty + 0.01 * l_reg
    else:
        gw = class_w[y]
        w_cell = 0.10 + 4.0 * gt.clamp(0, 1).pow(2) + 2.0 * (gt >= 0.50).to(gt.dtype)
        lh = (F.smooth_l1_loss(h, gt, reduction="none", beta=0.05) * w_cell * gw.view(-1, 1, 1)).mean()
        lp = (F.smooth_l1_loss(h.flatten(1).max(1).values, gt.flatten(1).max(1).values, reduction="none", beta=0.05) * gw).mean()
        lfo = focal_ce(out["aux_logits"].float(), y, class_w)
        lrk = ranking_stop_vs_safe(h.flatten(1).max(1).values, y, 0.20)
        lfl = stop_floor(h, gt, 0.70)
        parts.update(L_h=lh, L_peak=lp, L_focal=lfo, L_rank=lrk, L_floor=lfl)
        total = None  # weighted in trainer via phase
        parts["_raw"] = True
    parts["L_total_base"] = total if total is not None else h.new_zeros(())
    return parts, total, y
