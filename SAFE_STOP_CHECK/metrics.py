"""Spatial leeway + temporal intent → per-frame and clip SAFE/STOP scores.

Model-agnostic: heatmaps in, scores out. No checkpoint or student class here.

STOP spatial (GT-STOP frames)
  coverage = clip(max(pred[allowed]) / STOP_PEAK, 0, 1)   # 1 if peak in neighborhood >= 0.60
  leak     = fraction of BLUE cells with pred >= 0.50      # false hot off the dilated blob
  spatial  = clip(coverage * (1 - leak), 0, 1)
  If the allowed mask is empty (should not happen on STOP), spatial = 0.

SAFE spatial (GT-SAFE frames)
  peak_term = clip(1 - relu(max(pred) - 0.25) / 0.35, 0, 1)  # 1 at max<=0.25, 0 at max>=0.60
  frac      = fraction of cells with pred >= 0.25
  spatial   = peak_term * (1 - frac)

Intent
  series of length 6: peak in the allowed neighborhood (or the full grid if none).
  slope = last - first. sign with deadzone 0.05.
  1 if same sign or both flat; 0 if opposite; 0.5 if one flat and one moving.

quality = spatial_weight * spatial + intent_weight * intent, clipped to [0, 1].
Clip SAFE/STOP are means of quality on GT-SAFE / GT-STOP frames only. THREAT is not averaged in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from config import (
    CHEBYSHEV_RADIUS,
    HOT_THRESHOLD,
    INTENT_DEADZONE,
    INTENT_FRAME_OFFSETS,
    INTENT_WEIGHT,
    SAFE_FALSE_HOT_THRESHOLD,
    SAFE_PEAK,
    SAFE_PEAK_PENALTY_SPAN,
    SPATIAL_WEIGHT,
    STOP_FALSE_HOT_THRESHOLD,
    STOP_PEAK,
    HORIZON_1S_FRAMES,
    T_MIN,
)

SAFE, THREAT, STOP = "SAFE", "THREAT", "STOP"


def classify_peak(peak: float) -> str:
    """Current code cuts: SAFE < 0.25, THREAT [0.25, 0.60), STOP >= 0.60."""
    p = float(peak)
    if p >= STOP_PEAK:
        return STOP
    if p >= SAFE_PEAK:
        return THREAT
    return SAFE


def red_mask(heatmap: np.ndarray, threshold: float = HOT_THRESHOLD) -> np.ndarray:
    return np.asarray(heatmap, dtype=np.float32) >= float(threshold)


def dilate_chebyshev(mask: np.ndarray, radius: int = CHEBYSHEV_RADIUS) -> np.ndarray:
    """Morphological dilation with a (2r+1)×(2r+1) square = Chebyshev distance <= r."""
    m = np.asarray(mask, dtype=bool)
    r = int(radius)
    if r <= 0:
        return m.copy()
    h, w = m.shape
    out = np.zeros_like(m)
    ys, xs = np.nonzero(m)
    for y, x in zip(ys, xs):
        out[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)] = True
    return out


def allowed_mask(gt: np.ndarray, hot_threshold: float = HOT_THRESHOLD, radius: int = CHEBYSHEV_RADIUS) -> np.ndarray:
    """GT-red cells plus Chebyshev neighborhood. Student heat here is not a spatial error."""
    return dilate_chebyshev(red_mask(gt, hot_threshold), radius)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def spatial_stop(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    allowed = allowed_mask(gt)
    if not bool(allowed.any()):
        return {"spatial": 0.0, "coverage": 0.0, "leak": 1.0, "n_allowed": 0}
    coverage = _clip01(float(pred[allowed].max()) / float(STOP_PEAK))
    blue = ~allowed
    n_blue = int(blue.sum())
    leak = float(np.mean(pred[blue] >= STOP_FALSE_HOT_THRESHOLD)) if n_blue else 0.0
    return {
        "spatial": _clip01(coverage * (1.0 - leak)),
        "coverage": coverage,
        "leak": leak,
        "n_allowed": int(allowed.sum()),
    }


def spatial_safe(pred: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float32)
    peak = float(pred.max())
    peak_term = _clip01(1.0 - max(0.0, peak - SAFE_PEAK) / float(SAFE_PEAK_PENALTY_SPAN))
    frac = float(np.mean(pred >= SAFE_FALSE_HOT_THRESHOLD))
    return {
        "spatial": _clip01(peak_term * (1.0 - frac)),
        "coverage": peak_term,
        "leak": frac,
        "n_allowed": 0,
    }


def slope_sign(series: np.ndarray, deadzone: float = INTENT_DEADZONE) -> int:
    """+1 rising, -1 falling, 0 flat. slope = last - first (robust to mid-series noise)."""
    s = np.asarray(series, dtype=np.float32).ravel()
    if s.size < 2:
        return 0
    delta = float(s[-1] - s[0])
    if delta > deadzone:
        return 1
    if delta < -deadzone:
        return -1
    return 0


def intent_match(gt_series: np.ndarray, pred_series: np.ndarray, deadzone: float = INTENT_DEADZONE) -> float:
    g = slope_sign(gt_series, deadzone)
    p = slope_sign(pred_series, deadzone)
    if g == p:
        return 1.0
    if g == 0 or p == 0:
        return 0.5
    return 0.0


def _peak_in_region(heatmap: np.ndarray, region: np.ndarray | None) -> float:
    hm = np.asarray(heatmap, dtype=np.float32)
    if region is None or not bool(np.asarray(region).any()):
        return float(hm.max())
    return float(hm[np.asarray(region, dtype=bool)].max())


def intent_series(
    gt_maps: dict[int, np.ndarray],
    pred_maps: dict[int, np.ndarray],
    t: int,
    *,
    region_from_gt: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Aligned peaks: student at tau vs GT at tau+1s. Region is allowed mask of GT[tau+1s]."""
    g_vals: list[float] = []
    p_vals: list[float] = []
    for off in INTENT_FRAME_OFFSETS:
        tau = int(t) + int(off)
        if tau not in pred_maps or tau < T_MIN:
            continue
        tgt = tau + HORIZON_1S_FRAMES
        if tgt not in gt_maps:
            continue
        gt = gt_maps[tgt]
        region = allowed_mask(gt) if region_from_gt else None
        g_vals.append(_peak_in_region(gt, region))
        p_vals.append(_peak_in_region(pred_maps[tau], region))
    return np.asarray(g_vals, dtype=np.float32), np.asarray(p_vals, dtype=np.float32)


def per_frame_quality(
    gt: np.ndarray,
    pred: np.ndarray,
    gt_series: np.ndarray,
    pred_series: np.ndarray,
) -> dict[str, Any]:
    gt_cls = classify_peak(float(np.max(gt)))
    pred_cls = classify_peak(float(np.max(pred)))
    if gt_cls == SAFE:
        spat = spatial_safe(pred)
    else:
        spat = spatial_stop(pred, gt)
    if gt_series.size >= 2 and pred_series.size >= 2:
        intent = intent_match(gt_series, pred_series)
    else:
        intent = 0.5
    quality = _clip01(SPATIAL_WEIGHT * spat["spatial"] + INTENT_WEIGHT * intent)
    return {
        "gt_class": gt_cls,
        "pred_class": pred_cls,
        "gt_peak": float(np.max(gt)),
        "pred_peak": float(np.max(pred)),
        "spatial": float(spat["spatial"]),
        "coverage": float(spat["coverage"]),
        "leak": float(spat["leak"]),
        "intent": float(intent),
        "quality": quality,
    }


def clip_scores(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean quality on GT-SAFE and GT-STOP frames. THREAT is reported, not mixed in."""
    safe = [f for f in frames if f["gt_class"] == SAFE]
    stop = [f for f in frames if f["gt_class"] == STOP]
    threat = [f for f in frames if f["gt_class"] == THREAT]
    scored = frames

    def _mean(rows: list[dict[str, Any]], key: str) -> float | None:
        if not rows:
            return None
        return float(np.mean([r[key] for r in rows]))

    return {
        "n_frames": len(scored),
        "n_safe_frames": len(safe),
        "n_stop_frames": len(stop),
        "n_threat_frames": len(threat),
        "safe_score": _mean(safe, "quality"),
        "stop_score": _mean(stop, "quality"),
        "mean_spatial": _mean(scored, "spatial"),
        "mean_intent": _mean(scored, "intent"),
        "mean_spatial_safe": _mean(safe, "spatial"),
        "mean_spatial_stop": _mean(stop, "spatial"),
        "mean_intent_safe": _mean(safe, "intent"),
        "mean_intent_stop": _mean(stop, "intent"),
    }


def score_clip(
    gt_all: np.ndarray,
    pred_by_t: dict[int, np.ndarray],
    *,
    t_min: int = T_MIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score every t with a student pred and a GT map at t+1s."""
    n = int(gt_all.shape[0])
    gt_maps = {i: gt_all[i] for i in range(n)}
    rows: list[dict[str, Any]] = []
    for t in sorted(pred_by_t):
        if t < t_min:
            continue
        tgt = t + HORIZON_1S_FRAMES
        if tgt >= n:
            continue
        gt = gt_all[tgt]
        pred = pred_by_t[t]
        region_from_gt = classify_peak(float(np.max(gt))) != SAFE
        g_s, p_s = intent_series(gt_maps, pred_by_t, t, region_from_gt=region_from_gt)
        row = per_frame_quality(gt, pred, g_s, p_s)
        row["t"] = int(t)
        row["gt_t"] = int(tgt)
        rows.append(row)
    return rows, clip_scores(rows)


def fmt_score(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"
