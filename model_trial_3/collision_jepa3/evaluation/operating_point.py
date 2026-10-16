"""Cell-level operating-point metrics for H_1s (recall at fixed FA, AP, peak mean)."""

from __future__ import annotations

import numpy as np


def _valid_cells(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=np.float64).ravel()
    g = np.asarray(gt, dtype=np.float64).ravel()
    m = np.asarray(valid, dtype=bool).ravel()
    if m.shape != p.shape:
        m = np.broadcast_to(m, p.shape)
    return p[m], g[m]


def recall_at_false_alarm(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray,
    *,
    fa_rate: float = 0.05,
    hi: float = 0.5,
) -> dict[str, float]:
    p, g = _valid_cells(pred, gt, valid)
    pos = g >= hi
    neg = ~pos
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    out = {"n_pos": float(n_pos), "n_neg": float(n_neg), "fa_target": float(fa_rate)}
    if n_pos == 0:
        out.update(recall_at_fa=float("nan"), threshold=float("nan"), fa_achieved=float("nan"))
        return out
    if n_neg == 0:
        out.update(recall_at_fa=1.0, threshold=float(p.min()) if p.size else 0.0, fa_achieved=0.0)
        return out
    t = float(np.quantile(p[neg], 1.0 - fa_rate))
    fa = float(np.mean(p[neg] >= t))
    rec = float(np.mean(p[pos] >= t))
    out.update(recall_at_fa=rec, threshold=t, fa_achieved=fa)
    return out


def average_precision_high(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, hi: float = 0.5) -> float:
    p, g = _valid_cells(pred, gt, valid)
    y = (g >= hi).astype(np.int32)
    if y.size == 0 or int(y.sum()) == 0 or int(y.sum()) == int(y.size):
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, p))


def mean_pred_on_high(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, hi: float = 0.8) -> float:
    p, g = _valid_cells(pred, gt, valid)
    sel = g >= hi
    if int(sel.sum()) == 0:
        return float("nan")
    return float(p[sel].mean())


def unweighted_mse(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p, g = _valid_cells(pred, gt, valid)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - g) ** 2))


def fa_and_recall_at_threshold(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, thr: float, hi: float = 0.5) -> dict[str, float]:
    p, g = _valid_cells(pred, gt, valid)
    pos = g >= hi
    neg = ~pos
    rec = float(np.mean(p[pos] >= thr)) if int(pos.sum()) else float("nan")
    fa = float(np.mean(p[neg] >= thr)) if int(neg.sum()) else float("nan")
    return {"recall_at_thr": rec, "fa_at_thr": fa, "threshold": float(thr)}


def operating_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, hi: float = 0.5) -> dict[str, float]:
    rec5 = recall_at_false_alarm(pred, gt, valid, fa_rate=0.05, hi=hi)
    at05 = fa_and_recall_at_threshold(pred, gt, valid, 0.5, hi=hi)
    return {
        "recall_at_5fa": rec5["recall_at_fa"],
        "threshold_5fa": rec5["threshold"],
        "fa_achieved_5fa": rec5["fa_achieved"],
        "n_pos": rec5["n_pos"],
        "n_neg": rec5["n_neg"],
        "ap_high": average_precision_high(pred, gt, valid, hi=hi),
        "mean_pred_gt80": mean_pred_on_high(pred, gt, valid, hi=0.8),
        "mse": unweighted_mse(pred, gt, valid),
        "recall_at_0.5": at05["recall_at_thr"],
        "fa_at_0.5": at05["fa_at_thr"],
    }
