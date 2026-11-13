"""Short pilot training loops (≤8 epochs) for student research."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from collision_jepa3.student_research.cnn import StudentRGBNet
from collision_jepa3.student_research.pilot_metrics import accumulate_pilot_metrics


def _loss_gt_kd(out, batch, alpha: float) -> tuple[torch.Tensor, dict[str, float]]:
    gt = batch["gt_1s"]
    h = out["Hhat_1s"]
    l_gt = F.mse_loss(h, gt)
    l_kd = F.mse_loss(h, batch["H_teacher"])
    total = l_gt + alpha * l_kd
    return total, {"L_gt": float(l_gt.detach()), "L_kd": float(l_kd.detach()), "L_total": float(total.detach())}


def _loss_latent(out, batch, eta: float, mode: str = "smoothl1") -> torch.Tensor:
    z_s = out["Z_t"]
    z_t = batch["Z_teacher"]
    if mode == "cosine":
        ps = F.normalize(z_s.flatten(2), dim=1)
        pt = F.normalize(z_t.flatten(2), dim=1)
        return eta * (1.0 - (ps * pt).sum(dim=1).mean())
    return eta * F.smooth_l1_loss(z_s, z_t)


def run_pilot(
    model: StudentRGBNet,
    train_loader,
    val_loader,
    *,
    device: str,
    epochs: int,
    lr: float,
    alpha_kd: float = 0.5,
    eta_latent: float = 0.0,
    latent_mode: str = "smoothl1",
    cfg_eval,
    start_epoch: int = 0,
    on_epoch_end: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.trainable_parameters() if hasattr(model, "trainable_parameters") else model.parameters(), lr=lr)
    history = []
    best = None
    for ep in range(start_epoch, epochs):
        model.train()
        t0 = time.perf_counter()
        running = {k: 0.0 for k in ("L_total", "L_gt", "L_kd")}
        n = 0
        for batch in train_loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            out = model(batch["clip"])
            loss, terms = _loss_gt_kd(out, batch, alpha_kd)
            if eta_latent > 0:
                lz = _loss_latent(out, batch, eta_latent, latent_mode)
                loss = loss + lz
                terms["L_latent"] = float(lz.detach())
            loss.backward()
            opt.step()
            for k, v in terms.items():
                running[k] = running.get(k, 0.0) + v
            n += 1
        train_means = {k: v / max(n, 1) for k, v in running.items()}
        warning = cfg_eval.get("warning") or {}
        val = accumulate_pilot_metrics(model, val_loader, device, warning)
        rec = {"epoch": ep + 1, "train": train_means, "val": val, "seconds": time.perf_counter() - t0}
        history.append(rec)
        score = val.get("stop_recall")
        if best is None or (score is not None and score == score and score > best.get("stop_recall", -1)):
            best = {"epoch": ep + 1, **val}
        if on_epoch_end is not None:
            on_epoch_end(ep + 1, rec)
    return {"history": history, "best": best, "n_steps_per_epoch": n}
