"""Batch movement, finite checks, and now-map checkpoint selection."""

from __future__ import annotations

from typing import Any

import torch

from ..config import supervised_horizon_names


def teacher_forward(model, batch):
    """Use cached encoder tokens when present; otherwise run the backbone on frames."""
    if "tokens" in batch:
        return model(tokens=batch["tokens"])
    return model(clip=batch["frames"])


def move_batch(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN/Inf")


def current_lrs(optimizer) -> list[float]:
    return [float(g["lr"]) for g in optimizer.param_groups]


def horizon_mse(val: dict[str, Any], cfg) -> float:
    """Unweighted heatmap MSE of the supervised now map."""
    names = supervised_horizon_names(cfg)
    horizons = val.get("horizons") or {}
    vals: list[float] = []
    for n in names:
        raw = (horizons.get(n) or {}).get("mse")
        if raw is None:
            continue
        m = float(raw)
        if m == m:
            vals.append(m)
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def selection_better(cur: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Prefer an eligible now-map score, then the numeric score."""
    if best is None:
        return True
    c_el = bool(cur.get("eligible", False))
    b_el = bool(best.get("eligible", False))
    if c_el != b_el:
        return c_el
    return float(cur.get("score", float("-inf"))) > float(best.get("score", float("-inf")))


def composite_score(metrics: dict[str, Any], cfg) -> dict[str, Any]:
    """Score the now map: high-threat recall + STOP F1, with nuisance and miss gates."""
    names = supervised_horizon_names(cfg)
    hw = cfg.get("loss.horizon_weights")
    wsum = sum(float(hw[n]) for n in names) or 1.0
    rec = 0.0
    stop_f1 = 0.0
    nuisance = 0.0
    miss = 0.0
    for n in names:
        block = metrics.get(n, {})
        w = float(hw[n]) / wsum
        rec += w * float(block.get("high_threat_recall", 0.0))
        stop_f1 += w * float(block.get("stop_f1", 0.0))
        nuisance += w * float(block.get("nuisance", 0.0))
        miss += w * float(block.get("miss", 0.0))
    rec_w = float(cfg.get("selection.future_recall_weight", 0.6))
    stop_w = float(cfg.get("selection.stop_f1_weight", 0.4))
    nui_w = float(cfg.get("selection.nuisance_weight", 0.5))
    miss_w = float(cfg.get("selection.miss_weight", 0.5))
    score = rec_w * rec + stop_w * stop_f1 - nui_w * nuisance - miss_w * miss
    eligible = nuisance <= float(cfg.get("selection.max_nuisance", 0.20)) and miss <= float(
        cfg.get("selection.max_miss", 0.25)
    )
    return {
        "score": float(score),
        "eligible": bool(eligible),
        "recall": float(rec),
        "stop_f1": float(stop_f1),
        "nuisance": float(nuisance),
        "miss": float(miss),
    }
