"""M1 heatmap-dominant balancing: EMA group mix + adaptive STOP-floor."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from collision_jepa3.student_research.overnight6.losses import (
    STOP,
    occupied_heatmap,
    peak_class_t,
    selective_kd,
)

EPS = 1e-6
EMA_DECAY = 0.90
MULT_CLAMP = (0.05, 20.0)
FLOOR_TARGET = 0.70
FLOOR_BINS = [
    (0.00, 0.25, 4.0),
    (0.25, 0.45, 3.0),
    (0.45, 0.60, 2.0),
    (0.60, 0.70, 1.0),
    (0.70, 10.0, 0.0),
]


def task_ratio(epoch: int) -> float:
    table = {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.30, 5: 0.40}
    return float(table.get(int(epoch), 0.50))


def latent_group(lf, lc, lm, lr):
    return lf + 0.2 * lc + 0.5 * lm + 0.2 * lr


def task_group(lh, lfl, lkd):
    return 0.8 * lh + 0.6 * lfl + 0.4 * lkd


def floor_weight_from_peak(p: torch.Tensor) -> torch.Tensor:
    w = torch.zeros_like(p)
    w = torch.where(p < 0.25, torch.full_like(p, 4.0), w)
    w = torch.where((p >= 0.25) & (p < 0.45), torch.full_like(p, 3.0), w)
    w = torch.where((p >= 0.45) & (p < 0.60), torch.full_like(p, 2.0), w)
    w = torch.where((p >= 0.60) & (p < 0.70), torch.full_like(p, 1.0), w)
    w = torch.where(p >= 0.70, torch.zeros_like(p), w)
    return w


def adaptive_stop_floor(h: torch.Tensor, gt: torch.Tensor, target: float = FLOOR_TARGET):
    y = peak_class_t(gt)
    p = h.flatten(1).max(1).values
    m = y == STOP
    bins = {f"n_{a}_{b}": 0 for a, b, _ in FLOOR_BINS}
    bins.update({f"pen_{a}_{b}": 0.0 for a, b, _ in FLOOR_BINS})
    if not bool(m.any()):
        return h.new_zeros(()), bins
    ps = p[m]
    w = floor_weight_from_peak(ps.detach())
    pen = F.relu(target - ps)
    den = w.sum().clamp_min(1.0)
    loss = (w * pen).sum() / den
    with torch.no_grad():
        for a, b, _ in FLOOR_BINS:
            mm = (ps >= a) & (ps < b)
            bins[f"n_{a}_{b}"] = int(mm.sum())
            bins[f"pen_{a}_{b}"] = float(pen[mm].mean()) if bool(mm.any()) else 0.0
    return loss, bins


class GroupEMA:
    def __init__(self, decay: float = EMA_DECAY, eps: float = EPS, clamp: tuple[float, float] = MULT_CLAMP):
        self.decay = float(decay)
        self.eps = float(eps)
        self.clamp = clamp
        self.ema_lat = None
        self.ema_task = None

    def update(self, l_lat: torch.Tensor, l_task: torch.Tensor) -> None:
        a = max(float(l_lat.detach()), self.eps)
        b = max(float(l_task.detach()), self.eps)
        if self.ema_lat is None:
            self.ema_lat, self.ema_task = a, b
        else:
            d = self.decay
            self.ema_lat = d * self.ema_lat + (1.0 - d) * a
            self.ema_task = d * self.ema_task + (1.0 - d) * b

    def balance(self, l_lat: torch.Tensor, l_task: torch.Tensor, r: float):
        if self.ema_lat is None:
            self.update(l_lat, l_task)
        sl = (l_lat / (self.ema_lat + self.eps)).clamp(self.clamp[0], self.clamp[1])
        st = (l_task / (self.ema_task + self.eps)).clamp(self.clamp[0], self.clamp[1])
        t_lat = (1.0 - r) * sl
        t_task = r * st
        total = t_lat + t_task
        with torch.no_grad():
            s = float(total.detach()) + self.eps
            pct_task = float(t_task.detach()) / s
            pct_lat = float(t_lat.detach()) / s
        return total, pct_task, pct_lat


def m1_parts(out, batch, mu, sigma, predictor, decoder, dec_init):
    from collision_jepa3.student_research.overnight6.losses import decoder_anchor, latent_terms

    lf, lc, lr, lm = latent_terms(out, batch, mu, sigma, predictor)
    h = out["Hhat_1s"].float()
    gt = batch["gt_1s"].float()
    ht = batch["H_teacher"].float()
    lh = occupied_heatmap(h, gt)
    lfl, bins = adaptive_stop_floor(h, gt)
    lkd = selective_kd(h, ht, gt)
    lreg = decoder_anchor(decoder, dec_init)
    Llat = latent_group(lf, lc, lm, lr)
    Ltask = task_group(lh, lfl, lkd)
    parts = {
        "L_feat": lf,
        "L_cos": lc,
        "L_mgd": lm,
        "L_rel": lr,
        "L_h": lh,
        "L_floor": lfl,
        "L_kd": lkd,
        "L_dec_reg": lreg,
        "L_latent": Llat,
        "L_task": Ltask,
    }
    return parts, Llat, Ltask, lreg, bins
