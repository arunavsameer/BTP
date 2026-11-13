"""Warning metrics, μ+κσ risk, and zero-train oracles for U1."""

from __future__ import annotations

from typing import Any

import numpy as np

from .operating_point import operating_metrics

LABELS = ("SAFE", "THREAT", "STOP")
CAUTION = 0.25
STOP = 0.60
KAPPA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
STOP_SWEEP = (0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25)


def band(peak: np.ndarray, caution: float, stop: float) -> np.ndarray:
    out = np.zeros(np.asarray(peak).shape, dtype=np.int64)
    out[np.asarray(peak) >= float(caution)] = 1
    out[np.asarray(peak) >= float(stop)] = 2
    return out


def risk_maps(mu: np.ndarray, var: np.ndarray, kappa: float) -> np.ndarray:
    sigma = np.sqrt(np.clip(np.asarray(var, dtype=np.float64), 0.0, None))
    r = np.asarray(mu, dtype=np.float64) + float(kappa) * sigma
    return np.clip(r, 0.0, 1.0).astype(np.float32)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int = 3) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()):
        cm[int(t), int(p)] += 1
    return cm


def _cm_report(cm: np.ndarray) -> dict[str, Any]:
    rec: dict[str, float | None] = {}
    prec: dict[str, float | None] = {}
    for i, name in enumerate(LABELS):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        rec[name] = tp / (tp + fn) if (tp + fn) else None
        prec[name] = tp / (tp + fp) if (tp + fp) else None
    t_pos = int(cm[1:, :].sum())
    t_pos_hit = int(cm[1:, 1:].sum())
    safe_n = int(cm[0, :].sum())
    stop_n = int(cm[2, :].sum())
    return {
        "labels": list(LABELS),
        "counts_true_row_pred_col": cm.tolist(),
        "support": {name: int(cm[i, :].sum()) for i, name in enumerate(LABELS)},
        "recall": rec,
        "precision": prec,
        "threat_or_stop_recall": t_pos_hit / t_pos if t_pos else None,
        "safe_stop_rate": (int(cm[0, 2]) / safe_n) if safe_n else None,
        "stop_safe_count": int(cm[2, 0]),
        "stop_caught": int(cm[2, 2]),
        "n_stop_gt": stop_n,
        "n": int(cm.sum()),
    }


def _slice(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    caution: float,
    stop: float,
) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    if m.shape[0] != len(pred):
        raise ValueError("mask length mismatch")
    if not np.any(m):
        return {"n_frames": 0}
    p = np.asarray(pred, dtype=np.float32)[m]
    g = np.asarray(gt, dtype=np.float32)[m]
    gt_peak = g.reshape(len(g), -1).max(axis=1)
    pr_peak = p.reshape(len(p), -1).max(axis=1)
    gt_f = band(gt_peak, caution, stop)
    pr_f = band(pr_peak, caution, stop)
    op = operating_metrics(p, g, np.ones_like(p, dtype=bool))
    frame = _cm_report(confusion(gt_f, pr_f))
    cell = _cm_report(confusion(band(g, caution, stop), band(p, caution, stop)))
    return {
        "n_frames": int(len(p)),
        "n_cells": int(g.size),
        "gt_frame_mix": {n: int((gt_f == i).sum()) for i, n in enumerate(LABELS)},
        "operating": op,
        "frame": frame,
        "cell": cell,
        "recall_at_5fa": op.get("recall_at_5fa"),
        "mean_pred_gt80": op.get("mean_pred_gt80"),
        "stop_recall": frame["recall"]["STOP"],
        "safe_recall": frame["recall"]["SAFE"],
        "safe_stop_rate": frame["safe_stop_rate"],
        "stop_safe_count": frame["stop_safe_count"],
        "stop_caught": frame["stop_caught"],
        "mean_pred_peak": float(pr_peak.mean()),
        "mean_gt_peak": float(gt_peak.mean()),
    }


def warning_report(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pre_fork: np.ndarray,
    caution: float = CAUTION,
    stop: float = STOP,
    collision: np.ndarray | None = None,
    blob_n: np.ndarray | None = None,
    paired: np.ndarray | None = None,
) -> dict[str, Any]:
    pre = np.asarray(pre_fork, dtype=bool)
    n = len(pre)
    all_m = np.ones(n, dtype=bool)
    report = {
        "caution_threshold": float(caution),
        "stop_threshold": float(stop),
        "n_windows": int(n),
        "n_pre_fork": int(pre.sum()),
        "n_post_fork": int((~pre).sum()),
        "all": _slice(pred, gt, all_m, caution, stop),
        "pre_fork": _slice(pred, gt, pre, caution, stop),
        "post_fork": _slice(pred, gt, ~pre, caution, stop),
    }
    if collision is not None:
        coll = np.asarray(collision, dtype=bool)
        report["collision"] = _slice(pred, gt, coll, caution, stop)
        report["miss"] = _slice(pred, gt, ~coll, caution, stop)
    if blob_n is not None:
        small = pre & (np.asarray(blob_n) <= 2) & (np.asarray(gt).reshape(n, -1).max(axis=1) >= float(stop))
        report["small_blob_prefork_stop"] = _slice(pred, gt, small, caution, stop)
    if paired is not None:
        report["paired_prefork"] = _slice(pred, gt, pre & np.asarray(paired, dtype=bool), caution, stop)
        report["unpaired"] = _slice(pred, gt, ~np.asarray(paired, dtype=bool), caution, stop)
    a = report["all"]
    report["headline"] = {
        "stop_recall": a.get("stop_recall"),
        "safe_recall": a.get("safe_recall"),
        "pre_fork_stop_recall": report["pre_fork"].get("stop_recall"),
        "post_fork_stop_recall": report["post_fork"].get("stop_recall"),
        "recall_at_5fa": a.get("recall_at_5fa"),
        "safe_stop_rate": a.get("safe_stop_rate"),
        "stop_safe_count": a.get("stop_safe_count"),
        "stop_caught": a.get("stop_caught"),
        "small_blob_stop_recall": (report.get("small_blob_prefork_stop") or {}).get("stop_recall"),
    }
    return report


def feasible_vs_g(
    cand: dict[str, Any],
    g: dict[str, Any],
    *,
    rec_slack: float = 0.01,
    post_slack: float = 0.01,
    fa_slack: float = 0.01,
) -> dict[str, Any]:
    ch, gh = cand["headline"], g["headline"]

    def _f(x):
        return float(x) if x is not None and x == x else float("nan")

    def _ge(a, b, slack):
        fa, fb = _f(a), _f(b)
        if fa != fa or fb != fb:
            return True
        return fa >= fb - slack

    def _le(a, b, slack):
        fa, fb = _f(a), _f(b)
        if fa != fa or fb != fb:
            return True
        return fa <= fb + slack

    post_ok = _ge(ch["post_fork_stop_recall"], gh["post_fork_stop_recall"], post_slack)
    fa_ok = _le(ch["safe_stop_rate"], gh["safe_stop_rate"], fa_slack)
    rec_ok = _ge(ch["recall_at_5fa"], gh["recall_at_5fa"], rec_slack)
    return {
        "feasible": bool(post_ok and fa_ok and rec_ok),
        "post_fork_ok": bool(post_ok),
        "safe_stop_ok": bool(fa_ok),
        "rec5_ok": bool(rec_ok),
        "delta_pre_fork_stop": _f(ch["pre_fork_stop_recall"]) - _f(gh["pre_fork_stop_recall"]),
        "delta_stop": _f(ch["stop_recall"]) - _f(gh["stop_recall"]),
        "delta_safe": _f(ch["safe_recall"]) - _f(gh["safe_recall"]),
        "delta_rec5": _f(ch["recall_at_5fa"]) - _f(gh["recall_at_5fa"]),
        "delta_safe_stop": _f(ch["safe_stop_rate"]) - _f(gh["safe_stop_rate"]),
    }


def fit_mu_sigma_lookup(mu: np.ndarray, var: np.ndarray, mask: np.ndarray, n_bins: int = 20) -> dict[str, Any]:
    """E[v | μ] on masked cells. Used as the trivial ‘variance is a warp of the mean’ baseline."""
    m = np.asarray(mask, dtype=bool)
    if m.ndim == 1:
        m = np.broadcast_to(m[:, None, None], np.asarray(mu).shape)
    x = np.asarray(mu, dtype=np.float64)[m]
    y = np.asarray(var, dtype=np.float64)[m]
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means = np.zeros(int(n_bins), dtype=np.float64)
    counts = np.zeros(int(n_bins), dtype=np.int64)
    for i in range(int(n_bins)):
        sel = (x >= edges[i]) & (x < edges[i + 1] if i + 1 < int(n_bins) else x <= edges[i + 1])
        counts[i] = int(sel.sum())
        means[i] = float(y[sel].mean()) if counts[i] else 0.0
    filled = means.copy()
    last = 0.0
    for i in range(int(n_bins)):
        if counts[i] > 0:
            last = filled[i]
        else:
            filled[i] = last
    return {"edges": edges.tolist(), "centres": centres.tolist(), "means": filled.tolist(), "counts": counts.tolist()}


def apply_mu_sigma_lookup(mu: np.ndarray, lookup: dict[str, Any]) -> np.ndarray:
    edges = np.asarray(lookup["edges"], dtype=np.float64)
    means = np.asarray(lookup["means"], dtype=np.float64)
    x = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)
    idx = np.digitize(x, edges[1:-1], right=False)
    idx = np.clip(idx, 0, len(means) - 1)
    return means[idx].astype(np.float32)


def bernoulli_var(mu: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(mu, dtype=np.float64), 0.0, 1.0)
    return (x * (1.0 - x)).astype(np.float32)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def variance_diagnostics(v_hat: np.ndarray, v_true: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    win = np.asarray(mask, dtype=bool)
    if win.ndim != 1:
        win = win.reshape(len(v_hat), -1).any(axis=1)
    if not np.any(win):
        return {"n_cells": 0}
    ph = np.asarray(v_hat, dtype=np.float64)[win]
    pt = np.asarray(v_true, dtype=np.float64)[win]
    flat_h = ph.reshape(ph.shape[0], -1)
    flat_t = pt.reshape(pt.shape[0], -1)
    pos = pt.ravel() > 1e-4
    return {
        "n_cells": int(pt.size),
        "mae": float(np.mean(np.abs(ph - pt))),
        "mae_positive": float(np.mean(np.abs(pt.ravel()[pos] - ph.ravel()[pos]))) if np.any(pos) else None,
        "pearson": pearson(ph, pt),
        "mean_hat": float(ph.mean()),
        "mean_true": float(pt.mean()),
        "mean_hat_on_pos": float(ph.ravel()[pos].mean()) if np.any(pos) else None,
        "mean_hat_on_zero": float(ph.ravel()[~pos].mean()) if np.any(~pos) else None,
        "frac_hat_near_zero": float(np.mean(ph.ravel() < 1e-4)),
        "argmax_agree": float(np.mean(flat_h.argmax(axis=1) == flat_t.argmax(axis=1))),
        "n_positive_cells": int(pos.sum()),
    }


def upgrade_source(mu: np.ndarray, risk: np.ndarray, g_stop: float, u_stop: float) -> dict[str, Any]:
    """Among windows newly STOP under risk, did a different cell cause the peak?"""
    mu_peak = np.asarray(mu).reshape(len(mu), -1).max(axis=1)
    r_peak = np.asarray(risk).reshape(len(risk), -1).max(axis=1)
    new_stop = (r_peak >= float(u_stop)) & (mu_peak < float(g_stop))
    if not np.any(new_stop):
        return {"n_new_stop": 0, "frac_different_cell": None}
    arg_mu = np.asarray(mu).reshape(len(mu), -1)[new_stop].argmax(axis=1)
    arg_r = np.asarray(risk).reshape(len(risk), -1)[new_stop].argmax(axis=1)
    return {
        "n_new_stop": int(new_stop.sum()),
        "frac_different_cell": float(np.mean(arg_mu != arg_r)),
    }


def twin_accounting(
    pred_peak: np.ndarray,
    gt: np.ndarray,
    *,
    outcome_group: np.ndarray,
    anchor: np.ndarray,
    collision: np.ndarray,
    pre_fork: np.ndarray,
    paired: np.ndarray,
    caution: float = CAUTION,
    stop: float = STOP,
) -> dict[str, Any]:
    """Same prediction on both twins: collision STOP hits vs miss false STOP."""
    keys: dict[tuple[str, int], dict[str, int]] = {}
    gt_peak = np.asarray(gt).reshape(len(gt), -1).max(axis=1)
    pr = band(pred_peak, caution, stop)
    gt_b = band(gt_peak, caution, stop)
    coll = np.asarray(collision, dtype=bool)
    keep = np.asarray(pre_fork, dtype=bool) & np.asarray(paired, dtype=bool)
    for i in np.where(keep)[0]:
        key = (str(outcome_group[i]), int(anchor[i]))
        slot = keys.setdefault(key, {})
        if coll[i]:
            slot["c_pr"] = int(pr[i])
            slot["c_gt"] = int(gt_b[i])
        else:
            slot["m_pr"] = int(pr[i])
            slot["m_gt"] = int(gt_b[i])
    n = hit = miss_fp = both = 0
    for slot in keys.values():
        if "c_pr" not in slot or "m_pr" not in slot:
            continue
        n += 1
        if slot["c_gt"] == 2 and slot["c_pr"] == 2:
            hit += 1
        if slot["m_gt"] == 0 and slot["m_pr"] == 2:
            miss_fp += 1
        if slot["c_pr"] == 2 and slot["m_pr"] == 2:
            both += 1
    return {
        "n_complete_pairs": n,
        "collision_stop_hits": hit,
        "miss_safe_false_stop": miss_fp,
        "both_predicted_stop": both,
        "collision_stop_hit_rate": hit / n if n else None,
        "miss_false_stop_rate": miss_fp / n if n else None,
    }


def cluster_bootstrap(
    pred: np.ndarray,
    gt: np.ndarray,
    groups: np.ndarray,
    pre_fork: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 1337,
    caution: float = CAUTION,
    stop: float = STOP,
) -> dict[str, Any]:
    rng = np.random.RandomState(int(seed))
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_of = {u: np.where(g == u)[0] for u in uniq}
    keys = ("stop_recall", "safe_recall", "pre_fork_stop_recall", "recall_at_5fa", "safe_stop_rate")
    bag = {k: [] for k in keys}
    n_ok = 0
    for _ in range(int(n_boot)):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_of[p] for p in picks], axis=0)
        rep = warning_report(pred[idx], gt[idx], pre_fork=pre_fork[idx], caution=caution, stop=stop)
        h = rep["headline"]
        if any(h[k] is None or h[k] != h[k] for k in keys):
            continue
        n_ok += 1
        for k in keys:
            bag[k].append(float(h[k]))
    out: dict[str, Any] = {"n_boot": int(n_boot), "n_ok": n_ok, "n_groups": int(len(uniq))}
    for k, vals in bag.items():
        arr = np.asarray(vals, dtype=np.float64)
        if arr.size == 0:
            out[k] = None
            continue
        out[k] = {
            "mean": float(arr.mean()),
            "p05": float(np.quantile(arr, 0.05)),
            "p95": float(np.quantile(arr, 0.95)),
        }
    return out


def select_kappa(
    curves: list[dict[str, Any]],
    g_report: dict[str, Any],
) -> dict[str, Any]:
    feasible = [c for c in curves if c["vs_g"]["feasible"]]
    if not feasible:
        return {"kappa": 0.0, "reason": "no_feasible_kappa", "feasible_kappas": []}
    best = max(
        feasible,
        key=lambda c: (
            float(c["report"]["headline"]["pre_fork_stop_recall"] or -1),
            float(c["report"]["headline"]["recall_at_5fa"] or -1),
        ),
    )
    return {
        "kappa": float(best["kappa"]),
        "reason": "max_prefork_stop_among_feasible",
        "feasible_kappas": [float(c["kappa"]) for c in feasible],
        "vs_g": best["vs_g"],
        "headline": best["report"]["headline"],
    }


def jsonify(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and val != val:
            return None
        return val
    if isinstance(obj, float) and obj != obj:
        return None
    return obj
