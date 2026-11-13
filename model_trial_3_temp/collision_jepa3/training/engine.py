"""Batch movement, AMP, finite checks, and checkpoint selection."""

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
    """Mean unweighted heatmap MSE over supervised future horizons."""
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


def recall5_operating_valid(
    operating: dict[str, Any] | None,
    *,
    min_mean_gt80: float = 0.05,
    fa_lo: float = 0.03,
    fa_hi: float = 0.08,
) -> bool:
    """Reject collapsed all-zero maps whose rec@5FA is a ranking artifact."""
    if not operating:
        return False
    rec = operating.get("recall_at_5fa")
    fa = operating.get("fa_achieved_5fa")
    mean80 = operating.get("mean_pred_gt80")
    if rec is None or fa is None or mean80 is None:
        return False
    rec_f, fa_f, m_f = float(rec), float(fa), float(mean80)
    if rec_f != rec_f or fa_f != fa_f or m_f != m_f:
        return False
    if m_f < float(min_mean_gt80):
        return False
    if fa_f < float(fa_lo) or fa_f > float(fa_hi):
        return False
    return True


def recall5_better(operating: dict[str, Any] | None, best: dict[str, Any] | None) -> bool:
    """Global-best rec@5FA, only when the 5% FA threshold is well-defined."""
    if not recall5_operating_valid(operating):
        return False
    rec = float(operating["recall_at_5fa"])
    if best is None:
        return True
    prev = best.get("recall_at_5fa")
    if prev is None:
        return True
    prev_f = float(prev)
    if prev_f != prev_f:
        return True
    return rec > prev_f


def prefork_stop_better(
    h1s: dict[str, Any] | None,
    best: dict[str, Any] | None,
    *,
    post_floor: float = 0.90,
) -> bool:
    """Improve pre-fork STOP recall without letting post-fork STOP fall below a floor."""
    if not h1s:
        return False
    pre = h1s.get("pre_fork_stop_recall")
    post = h1s.get("post_fork_stop_recall")
    if pre is None or post is None:
        return False
    pre_f, post_f = float(pre), float(post)
    if pre_f != pre_f or post_f != post_f:
        return False
    floor = float(post_floor)
    if best is None:
        return True
    best_pre = best.get("pre_fork_stop_recall")
    best_post = best.get("post_fork_stop_recall")
    best_pre_f = float(best_pre) if best_pre is not None else float("nan")
    best_post_f = float(best_post) if best_post is not None else float("nan")
    if post_f < floor:
        if best_post_f != best_post_f or best_post_f < floor:
            return post_f > (best_post_f if best_post_f == best_post_f else float("-inf"))
        return False
    if best_post_f != best_post_f or best_post_f < floor:
        return True
    if best_pre_f != best_pre_f:
        return True
    return pre_f > best_pre_f


def _finite(x) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return v == v


def balanced_warning_better(
    h1s: dict[str, Any] | None,
    best: dict[str, Any] | None,
    *,
    post_floor: float = 0.90,
    safe_floor: float = 0.82,
    stop_floor: float = 0.75,
    max_safe_stop: float = 0.08,
) -> bool:
    """Raise min(SAFE, STOP) recall without a STOP false-alarm spike.

    Rejects maps that miss the post-fork STOP floor, drop below SAFE/STOP
    floors once they have been met, or exceed the SAFE→STOP rate cap.
    """
    if not h1s:
        return False
    safe = h1s.get("safe_recall")
    stop = h1s.get("stop_recall")
    post = h1s.get("post_fork_stop_recall")
    fa = h1s.get("safe_stop_rate")
    if not all(_finite(x) for x in (safe, stop, post, fa)):
        return False
    safe_f, stop_f, post_f, fa_f = float(safe), float(stop), float(post), float(fa)
    score = min(safe_f, stop_f)
    if best is None:
        return True
    best_safe = best.get("safe_recall")
    best_stop = best.get("stop_recall")
    best_post = best.get("post_fork_stop_recall")
    best_fa = best.get("safe_stop_rate")
    best_score = (
        min(float(best_safe), float(best_stop))
        if _finite(best_safe) and _finite(best_stop)
        else float("-inf")
    )
    best_post_f = float(best_post) if _finite(best_post) else float("nan")
    best_fa_f = float(best_fa) if _finite(best_fa) else float("inf")
    if post_f < float(post_floor):
        if not _finite(best_post) or best_post_f < float(post_floor):
            return post_f > (best_post_f if _finite(best_post) else float("-inf"))
        return False
    if fa_f > float(max_safe_stop) and _finite(best_fa) and best_fa_f <= float(max_safe_stop):
        return False
    if _finite(best_safe) and float(best_safe) >= float(safe_floor) and safe_f < float(safe_floor):
        return False
    if _finite(best_stop) and float(best_stop) >= float(stop_floor) and stop_f < float(stop_floor):
        return False
    return score > best_score


def selection_better(cur: dict[str, Any], best: dict[str, Any] | None) -> bool:
    """Prefer eligible warning quality, then the documented composite score."""
    if best is None:
        return True
    c_el = bool(cur.get("eligible", False))
    b_el = bool(best.get("eligible", False))
    if c_el != b_el:
        return c_el
    return float(cur.get("score", float("-inf"))) > float(best.get("score", float("-inf")))


def composite_score(metrics: dict[str, Any], cfg) -> dict[str, Any]:
    """Score from future high-threat recall + STOP F1, with nuisance/miss gates."""
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
        "future_recall": float(rec),
        "stop_f1": float(stop_f1),
        "nuisance": float(nuisance),
        "miss": float(miss),
    }
