"""Per-channel ICC from matched-background pairs (frozen encoder features)."""

from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-8


def anchor_feature_vec(tokens: np.ndarray) -> np.ndarray:
    """Pool one anchor window ``[n_tokens, C]`` (or ``[T, n_tokens, C]``) to ``[C]``."""
    x = np.asarray(tokens, dtype=np.float64)
    if x.ndim == 3:
        return x.mean(axis=(0, 1))
    if x.ndim == 2:
        return x.mean(axis=0)
    raise ValueError(f"expected 2D or 3D tokens, got {x.shape}")


def pair_w_b_icc(
    fa: np.ndarray,
    fb: np.ndarray,
    eps: float = DEFAULT_EPS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``fa``, ``fb``: ``[N, C]`` pair endpoints. Returns ``W``, ``B``, ``ICC`` each ``[C]``."""
    if fa.shape != fb.shape or fa.ndim != 2:
        raise ValueError(f"expected matching [N, C], got {fa.shape} and {fb.shape}")
    diff = fa - fb
    w = 0.5 * (diff * diff).mean(axis=0)
    mid = 0.5 * (fa + fb)
    b = np.maximum(mid.var(axis=0) - 0.5 * w, 0.0)
    icc = b / (b + w + float(eps))
    return w, b, icc


def icc_percentiles(icc: np.ndarray, ps: tuple[float, ...] = (10, 25, 50, 75, 90)) -> dict[str, float]:
    return {f"p{int(p)}": float(np.percentile(icc, p)) for p in ps}


def select_low_icc_channels(
    icc: np.ndarray,
    w: np.ndarray,
    *,
    w_quantile: float = 0.5,
    low_frac: float = 0.2,
) -> dict[str, object]:
    """Eligible if ``W_c >= median(W)``; return lowest-ICC ``low_frac`` among eligible."""
    w_thr = float(np.median(w))
    eligible = w >= w_thr
    n_eligible = int(eligible.sum())
    out: dict[str, object] = {
        "w_median_threshold": w_thr,
        "n_channels": int(icc.shape[0]),
        "n_eligible": n_eligible,
        "low_frac": float(low_frac),
        "selected_channel_ids": [],
    }
    if n_eligible == 0:
        return out
    sub_icc = icc[eligible]
    sub_ids = np.nonzero(eligible)[0]
    k = max(1, int(np.ceil(low_frac * n_eligible)))
    order = np.argsort(sub_icc)
    pick = sub_ids[order[:k]]
    out["n_selected"] = int(k)
    out["selected_channel_ids"] = [int(i) for i in pick.tolist()]
    out["selected_icc_min"] = float(sub_icc[order[0]])
    out["selected_icc_max"] = float(sub_icc[order[k - 1]])
    return out
