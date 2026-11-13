"""Honest model-judge metrics for 5×5 H_1s maps.

The training/val-score criteria are a bad way to pick a winner:

* Composite warning score used to mix row weights and hardcoded 0.42/0.70 cuts.
* In-sample recall@5% FA fits the threshold on the same cells it scores, and
  with tied scores the quantile rule can report recall=1 at a 100% FA rate.
* Pooled cell AP lets one high-threat clip dominate the number.
* Unweighted MSE is dominated by empty cells; mean pred on GT>=0.8 rewards
  over-confident maps with no false-alarm cost.

This module scores independent families (ranking, nested FA, localization,
calibration, collapse, robustness) and aggregates eligible models with Borda
ranks plus clip-bootstrap comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .metrics import classify_heatmap
from .operating_point import recall_at_false_alarm

GRID = 5
HI = 0.5
PEAK_HI = 0.5
SAFE_CELL = 0.05
DEFAULT_WARNING = {
    "row_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    "left_cols": [0, 1],
    "center_cols": [2],
    "right_cols": [3, 4],
    "caution_threshold": 0.25,
    "stop_threshold": 0.60,
}
LOSS_ROW_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0]

# Higher is better unless noted. Used for Borda aggregation.
RANK_METRICS = (
    ("nested_rec_at_2fa", True),
    ("nested_rec_at_5fa", True),
    ("clip_macro_ap", True),
    ("peak_hit", True),
    ("high_iou", True),
    ("high_mae", False),
)


@dataclass
class HeatmapPack:
    pred: np.ndarray
    gt: np.ndarray
    valid: np.ndarray
    clip_id: np.ndarray
    scenario: np.ndarray | None = None
    background: np.ndarray | None = None
    outcome: np.ndarray | None = None
    pre_fork: np.ndarray | None = None
    anchor: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.pred = np.asarray(self.pred, dtype=np.float64)
        self.gt = np.asarray(self.gt, dtype=np.float64)
        if self.pred.shape != self.gt.shape or self.pred.ndim != 3:
            raise ValueError(f"pred/gt must be (N,5,5), got {self.pred.shape} {self.gt.shape}")
        n = int(self.pred.shape[0])
        self.valid = np.ones(n, dtype=bool) if self.valid is None else np.asarray(self.valid, dtype=bool).reshape(-1)
        if self.valid.shape[0] != n:
            raise ValueError("valid length mismatch")
        self.clip_id = np.asarray(self.clip_id).reshape(-1)
        if self.clip_id.shape[0] != n:
            raise ValueError("clip_id length mismatch")

    @property
    def n(self) -> int:
        return int(self.pred.shape[0])

    def copy_pred(self, pred: np.ndarray) -> "HeatmapPack":
        return HeatmapPack(
            pred=pred,
            gt=self.gt,
            valid=self.valid,
            clip_id=self.clip_id,
            scenario=self.scenario,
            background=self.background,
            outcome=self.outcome,
            pre_fork=self.pre_fork,
            anchor=self.anchor,
        )

    def take(self, idx: np.ndarray) -> "HeatmapPack":
        idx = np.asarray(idx, dtype=int).reshape(-1)
        def _opt(arr):
            return None if arr is None else np.asarray(arr)[idx]

        return HeatmapPack(
            pred=self.pred[idx],
            gt=self.gt[idx],
            valid=self.valid[idx],
            clip_id=self.clip_id[idx],
            scenario=_opt(self.scenario),
            background=_opt(self.background),
            outcome=_opt(self.outcome),
            pre_fork=_opt(self.pre_fork),
            anchor=_opt(self.anchor),
        )


def _finite(x: float) -> float:
    x = float(x)
    return x if np.isfinite(x) else float("nan")


def _cells(pack: HeatmapPack) -> tuple[np.ndarray, np.ndarray]:
    m = pack.valid
    return pack.pred[m].ravel(), pack.gt[m].ravel()


def _binary_pos(gt: np.ndarray, hi: float = HI) -> np.ndarray:
    return np.asarray(gt, dtype=np.float64) >= float(hi)


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks, 1 = smallest. NaN ranks last."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    ranks = np.full(n, np.nan, dtype=np.float64)
    order = np.argsort(np.where(np.isfinite(x), x, np.inf), kind="mergesort")
    i = 0
    while i < n:
        if not np.isfinite(x[order[i]]):
            ranks[order[i:]] = float(n)
            break
        j = i
        while j + 1 < n and np.isfinite(x[order[j + 1]]) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return float("nan")
    return float((a * b).sum() / den)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rankdata_average(np.asarray(a, dtype=np.float64).ravel()), _rankdata_average(np.asarray(b, dtype=np.float64).ravel()))


def _sklearn_ap(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    if y.size == 0 or int(y.sum()) == 0 or int(y.sum()) == int(y.size):
        return float("nan")
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def _sklearn_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    if y.size == 0 or int(y.sum()) == 0 or int(y.sum()) == int(y.size):
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, s))


def recall_at_fa_capped(
    pred: np.ndarray,
    gt: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    fa_rate: float = 0.05,
    hi: float = HI,
) -> dict[str, float]:
    """Max recall among thresholds with FA <= fa_rate (cell-level).

    Always includes a threshold above max(pred) so a constant map cannot claim
    recall=1 when every cell fires. If no threshold meets the FA cap besides
    'never fire', recall is 0.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    g = np.asarray(gt, dtype=np.float64).ravel()
    if valid is None:
        m = np.ones(p.shape[0], dtype=bool)
    else:
        m = np.asarray(valid, dtype=bool).ravel()
        if m.shape[0] != p.shape[0]:
            m = np.broadcast_to(np.asarray(valid, dtype=bool), pred.shape).ravel()
    p, g = p[m], g[m]
    pos = _binary_pos(g, hi)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    out = {"n_pos": float(n_pos), "n_neg": float(n_neg), "fa_target": float(fa_rate)}
    if n_pos == 0:
        out.update(recall=float("nan"), fa=float("nan"), threshold=float("nan"), feasible=False)
        return out
    if p.size == 0:
        out.update(recall=0.0, fa=0.0, threshold=float("inf"), feasible=False)
        return out
    if n_neg == 0:
        out.update(recall=1.0, fa=0.0, threshold=float(p.min()), feasible=True)
        return out
    order = np.argsort(-p, kind="mergesort")
    p_ord = p[order]
    pos_ord = pos[order]
    tp = np.cumsum(pos_ord)
    fp = np.cumsum(~pos_ord)
    last = np.r_[np.flatnonzero(p_ord[1:] != p_ord[:-1]), p_ord.size - 1]
    recs = np.r_[0.0, tp[last] / float(n_pos)]
    fas = np.r_[0.0, fp[last] / float(n_neg)]
    thrs = np.r_[float(p_ord[0] + 1.0), p_ord[last].astype(np.float64)]
    ok = fas <= float(fa_rate) + 1e-12
    if not bool(ok.any()):
        out.update(recall=0.0, fa=0.0, threshold=float(thrs[0]), feasible=True)
        return out
    idx = np.flatnonzero(ok)
    best_rec = float(recs[idx].max())
    cands = idx[recs[idx] >= best_rec - 1e-15]
    best = int(cands[np.argmin(fas[cands])])
    out.update(recall=float(recs[best]), fa=float(fas[best]), threshold=float(thrs[best]), feasible=True)
    return out


def _threshold_for_fa(pred: np.ndarray, gt: np.ndarray, fa_rate: float, hi: float = HI) -> float:
    res = recall_at_fa_capped(pred, gt, fa_rate=fa_rate, hi=hi)
    t = res["threshold"]
    return float(t) if np.isfinite(t) else float("inf")


def _clip_groups(clip_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    clips, inv = np.unique(np.asarray(clip_id).astype(str), return_inverse=True)
    groups = [np.flatnonzero(inv == i) for i in range(clips.size)]
    return clips, inv, groups


def nested_recall_at_fa(
    pack: HeatmapPack,
    *,
    fa_rate: float = 0.05,
    n_folds: int = 5,
    hi: float = HI,
    seed: int = 0,
) -> dict[str, float]:
    """Fit the FA-capped threshold on other clips, score held-out clips."""
    clips, inv, _groups = _clip_groups(pack.clip_id)
    n_folds = int(max(2, min(n_folds, clips.size)))
    rng = np.random.RandomState(seed)
    order = rng.permutation(clips.size)
    folds = np.array_split(order, n_folds)
    recs: list[float] = []
    fas: list[float] = []
    valid = pack.valid
    for fold in folds:
        is_eval = np.zeros(clips.size, dtype=bool)
        is_eval[fold] = True
        eval_mask = is_eval[inv] & valid
        calib_mask = (~is_eval[inv]) & valid
        if int(eval_mask.sum()) == 0 or int(calib_mask.sum()) == 0:
            continue
        t = _threshold_for_fa(pack.pred[calib_mask], pack.gt[calib_mask], fa_rate, hi=hi)
        ev_p = pack.pred[eval_mask]
        ev_g = pack.gt[eval_mask]
        pos = _binary_pos(ev_g, hi)
        neg = ~pos
        if int(pos.sum()) == 0:
            continue
        recs.append(float(np.mean(ev_p[pos] >= t)))
        fas.append(float(np.mean(ev_p[neg] >= t)) if int(neg.sum()) else 0.0)
    if not recs:
        return {"recall": float("nan"), "fa": float("nan"), "n_folds": 0.0, "std": float("nan")}
    arr = np.asarray(recs, dtype=np.float64)
    return {
        "recall": float(arr.mean()),
        "fa": float(np.mean(fas)),
        "n_folds": float(len(recs)),
        "std": float(arr.std(ddof=0)),
    }


def clip_aps(pack: HeatmapPack, hi: float = HI) -> dict[str, float]:
    aps: list[float] = []
    for cid in sorted(set(pack.clip_id.tolist()), key=str):
        idx = (pack.clip_id == cid) & pack.valid
        if int(idx.sum()) == 0:
            continue
        y = _binary_pos(pack.gt[idx], hi).ravel()
        s = pack.pred[idx].ravel()
        ap = _sklearn_ap(y, s)
        if np.isfinite(ap):
            aps.append(ap)
    if not aps:
        return {"clip_macro_ap": float("nan"), "n_clips": 0.0, "p25": float("nan")}
    arr = np.asarray(aps, dtype=np.float64)
    return {
        "clip_macro_ap": float(arr.mean()),
        "n_clips": float(arr.size),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.median(arr)),
        "min": float(arr.min()),
    }


def weighted_mse(pack: HeatmapPack, row_weights: Iterable[float] | None = None) -> float:
    p, g = pack.pred, pack.gt
    w = 0.05 + 4.0 * np.clip(g, 0.0, None) ** 2
    if row_weights is not None:
        rw = np.asarray(list(row_weights), dtype=np.float64).reshape(1, -1, 1)
        w = w * rw
    m = pack.valid[:, None, None]
    w = np.where(m, w, 0.0)
    se = (p - g) ** 2
    denom = float(w.sum())
    if denom <= 1e-12:
        return float("nan")
    return float((se * w).sum() / denom)


def unweighted_mse(pack: HeatmapPack) -> float:
    p, g = _cells(pack)
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - g) ** 2))


def high_mae(pack: HeatmapPack, hi: float = HI) -> float:
    p, g = _cells(pack)
    sel = g >= hi
    if int(sel.sum()) == 0:
        return float("nan")
    return float(np.mean(np.abs(p[sel] - g[sel])))


def peak_metrics(pack: HeatmapPack, hi: float = PEAK_HI) -> dict[str, float]:
    hits = []
    dist = []
    height_err = []
    for i in range(pack.n):
        if not pack.valid[i]:
            continue
        g = pack.gt[i]
        if float(g.max()) < hi:
            continue
        p = pack.pred[i]
        g_idx = int(np.argmax(g))
        p_idx = int(np.argmax(p))
        gr, gc = divmod(g_idx, GRID)
        pr, pc = divmod(p_idx, GRID)
        hits.append(int(g_idx == p_idx))
        dist.append(float(max(abs(gr - pr), abs(gc - pc))))
        height_err.append(float(abs(float(p.max()) - float(g.max()))))
    if not hits:
        return {"peak_hit": float("nan"), "peak_chebyshev": float("nan"), "peak_height_mae": float("nan"), "n_peak": 0.0}
    return {
        "peak_hit": float(np.mean(hits)),
        "peak_chebyshev": float(np.mean(dist)),
        "peak_height_mae": float(np.mean(height_err)),
        "n_peak": float(len(hits)),
    }


def high_iou(pack: HeatmapPack, hi: float = HI) -> float:
    ious = []
    for i in range(pack.n):
        if not pack.valid[i]:
            continue
        g = pack.gt[i] >= hi
        if not bool(g.any()):
            continue
        p = pack.pred[i] >= hi
        inter = float(np.sum(p & g))
        union = float(np.sum(p | g))
        ious.append(inter / union if union else 0.0)
    return float(np.mean(ious)) if ious else float("nan")


def topk_overlap(pack: HeatmapPack, k: int = 3, hi: float = PEAK_HI) -> float:
    scores = []
    k = int(k)
    for i in range(pack.n):
        if not pack.valid[i]:
            continue
        g = pack.gt[i].ravel()
        if float(g.max()) < hi:
            continue
        p = pack.pred[i].ravel()
        g_top = set(np.argsort(g)[-k:].tolist())
        p_top = set(np.argsort(p)[-k:].tolist())
        scores.append(len(g_top & p_top) / float(k))
    return float(np.mean(scores)) if scores else float("nan")


def direction_accuracy(pack: HeatmapPack, warning: dict | None = None) -> float:
    warn = warning or DEFAULT_WARNING
    n = 0
    c = 0
    for i in range(pack.n):
        if not pack.valid[i]:
            continue
        t_dir, t_sev, _ = classify_heatmap(pack.gt[i], warn)
        if t_sev < 1:
            continue
        p_dir, _, _ = classify_heatmap(pack.pred[i], warn)
        n += 1
        c += int(p_dir == t_dir)
    return float(c / n) if n else float("nan")


def stop_ap(pack: HeatmapPack, warning: dict | None = None) -> float:
    warn = warning or DEFAULT_WARNING
    y = []
    s = []
    for i in range(pack.n):
        if not pack.valid[i]:
            continue
        _, t_sev, t_scores = classify_heatmap(pack.gt[i], warn)
        _, _, p_scores = classify_heatmap(pack.pred[i], warn)
        y.append(int(t_sev >= 2))
        s.append(float(max(p_scores.values())))
    return _sklearn_ap(np.asarray(y), np.asarray(s))


def brier_high(pack: HeatmapPack, hi: float = HI) -> float:
    p, g = _cells(pack)
    if p.size == 0:
        return float("nan")
    y = (g >= hi).astype(np.float64)
    s = np.clip(p, 0.0, 1.0)
    return float(np.mean((s - y) ** 2))


def spatial_std(maps: np.ndarray, valid: np.ndarray) -> float:
    if int(valid.sum()) == 0:
        return float("nan")
    return float(maps[valid].std(axis=(-2, -1)).mean())


def mean_cosine_to_map(pack: HeatmapPack, ref: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64).ravel()
    rn = np.linalg.norm(ref)
    if rn <= 1e-12:
        return float("nan")
    ref = ref / rn
    cos = []
    for i in range(pack.n):
        if not pack.valid[i]:
            continue
        v = pack.pred[i].ravel()
        vn = np.linalg.norm(v)
        if vn <= 1e-12:
            continue
        cos.append(float(np.dot(v / vn, ref)))
    return float(np.mean(cos)) if cos else float("nan")


def subgroup_ap(pack: HeatmapPack, labels: np.ndarray | None, hi: float = HI, min_n: int = 8) -> dict[str, float]:
    out: dict[str, float] = {}
    if labels is None:
        return out
    labels = np.asarray(labels)
    for lab in sorted(set(labels.tolist()), key=str):
        idx = np.array([i for i in range(pack.n) if pack.valid[i] and labels[i] == lab], dtype=int)
        if idx.size < min_n:
            continue
        y = _binary_pos(pack.gt[idx], hi).ravel()
        s = pack.pred[idx].ravel()
        out[str(lab)] = _sklearn_ap(y, s)
    return out


def score_pack(
    pack: HeatmapPack,
    *,
    warning: dict | None = None,
    seed: int = 0,
    n_folds: int = 5,
) -> dict[str, float]:
    """All judge metrics for one prediction pack."""
    p_cells, g_cells = _cells(pack)
    y = _binary_pos(g_cells)
    pooled_ap = _sklearn_ap(y.astype(np.int32), p_cells)
    pooled_auroc = _sklearn_auroc(y.astype(np.int32), p_cells)
    nested2 = nested_recall_at_fa(pack, fa_rate=0.02, n_folds=n_folds, seed=seed)
    nested5 = nested_recall_at_fa(pack, fa_rate=0.05, n_folds=n_folds, seed=seed)
    cell_valid = np.broadcast_to(pack.valid[:, None, None], pack.pred.shape)
    capped2 = recall_at_fa_capped(pack.pred, pack.gt, cell_valid, fa_rate=0.02)
    capped5 = recall_at_fa_capped(pack.pred, pack.gt, cell_valid, fa_rate=0.05)
    legacy5 = recall_at_false_alarm(pack.pred, pack.gt, cell_valid, fa_rate=0.05)
    clips = clip_aps(pack)
    peaks = peak_metrics(pack)
    high_p, high_g = p_cells[g_cells >= 0.8], g_cells[g_cells >= 0.8]
    scn = subgroup_ap(pack, pack.scenario)
    fork_labels = None
    if pack.pre_fork is not None:
        fork_labels = np.array(["pre" if bool(x) else "post" for x in pack.pre_fork], dtype=object)
    fork = subgroup_ap(pack, fork_labels, min_n=4)
    pred_mean = pack.pred[pack.valid].mean(axis=0) if int(pack.valid.sum()) else np.zeros((GRID, GRID))
    metrics = {
        "n_windows": float(pack.n),
        "n_valid": float(int(pack.valid.sum())),
        "n_clips": float(len(set(pack.clip_id.tolist()))),
        "pooled_ap": pooled_ap,
        "pooled_auroc": pooled_auroc,
        "clip_macro_ap": clips["clip_macro_ap"],
        "clip_ap_p25": clips.get("p25", float("nan")),
        "clip_ap_min": clips.get("min", float("nan")),
        "pearson": pearson(p_cells, g_cells),
        "spearman": spearman(p_cells, g_cells),
        "nested_rec_at_2fa": nested2["recall"],
        "nested_rec_at_2fa_fa": nested2["fa"],
        "nested_rec_at_2fa_std": nested2["std"],
        "nested_rec_at_5fa": nested5["recall"],
        "nested_rec_at_5fa_fa": nested5["fa"],
        "nested_rec_at_5fa_std": nested5["std"],
        "capped_rec_at_2fa": capped2["recall"],
        "capped_rec_at_5fa": capped5["recall"],
        "capped_fa_at_5fa": capped5["fa"],
        "legacy_rec_at_5fa": legacy5.get("recall_at_fa", float("nan")),
        "legacy_fa_achieved_5fa": legacy5.get("fa_achieved", float("nan")),
        "mse": unweighted_mse(pack),
        "weighted_mse": weighted_mse(pack, LOSS_ROW_WEIGHTS),
        "high_mae": high_mae(pack),
        "brier_high": brier_high(pack),
        "mean_pred_gt80": float(high_p.mean()) if high_p.size else float("nan"),
        "peak_hit": peaks["peak_hit"],
        "peak_chebyshev": peaks["peak_chebyshev"],
        "peak_height_mae": peaks["peak_height_mae"],
        "high_iou": high_iou(pack),
        "top3_overlap": topk_overlap(pack, k=3),
        "direction_acc": direction_accuracy(pack, warning),
        "stop_ap": stop_ap(pack, warning),
        "pred_spatial_std": spatial_std(pack.pred, pack.valid),
        "gt_spatial_std": spatial_std(pack.gt, pack.valid),
        "cosine_to_mean_pred": mean_cosine_to_map(pack, pred_mean),
        "scenario_ap_min": float(min(scn.values())) if scn else float("nan"),
        "scenario_ap_mean": float(np.mean(list(scn.values()))) if scn else float("nan"),
        "pre_fork_ap": fork.get("pre", float("nan")),
        "post_fork_ap": fork.get("post", float("nan")),
    }
    return {k: _finite(v) for k, v in metrics.items()}


def mean_heatmap_prior(gt: np.ndarray, valid: np.ndarray, like: HeatmapPack) -> HeatmapPack:
    m = valid.astype(bool)
    if int(m.sum()) == 0:
        mean = np.zeros((GRID, GRID), dtype=np.float64)
    else:
        mean = np.asarray(gt, dtype=np.float64)[m].mean(axis=0)
    pred = np.broadcast_to(mean, like.gt.shape).copy()
    return like.copy_pred(pred)


def gates_vs_prior(model: dict[str, float], prior: dict[str, float]) -> dict[str, Any]:
    """A useful model must beat the mean heatmap on ranking and weighted error,
    and must not collapse to a spatially flat map."""
    ap_ok = float(model.get("clip_macro_ap", 0.0)) > float(prior.get("clip_macro_ap", 0.0)) + 0.01
    wmse_ok = float(model.get("weighted_mse", 1e9)) < float(prior.get("weighted_mse", 0.0)) * 0.98
    gt_std = float(model.get("gt_spatial_std", 0.0))
    pred_std = float(model.get("pred_spatial_std", 0.0))
    spread_ok = (not np.isfinite(gt_std)) or gt_std <= 1e-8 or pred_std >= 0.30 * gt_std
    passed = bool(ap_ok and wmse_ok and spread_ok)
    return {
        "pass": passed,
        "beats_prior_clip_ap": bool(ap_ok),
        "beats_prior_weighted_mse": bool(wmse_ok),
        "not_spatially_collapsed": bool(spread_ok),
    }


def borda_ranks(scoreboard: dict[str, dict[str, float]], names: list[str]) -> dict[str, float]:
    """Mean competition rank (1=best) across RANK_METRICS. NaN ranks last."""
    acc = {n: [] for n in names}
    for key, higher in RANK_METRICS:
        vals = np.array([scoreboard[n].get(key, float("nan")) for n in names], dtype=np.float64)
        signed = np.where(np.isfinite(vals), -vals if higher else vals, np.inf)
        ranks = _rankdata_average(signed)
        for i, n in enumerate(names):
            acc[n].append(float(ranks[i]))
    return {n: float(np.mean(acc[n])) if acc[n] else float("nan") for n in names}


def _resample_from_groups(groups: list[np.ndarray], sampled: np.ndarray) -> np.ndarray:
    parts = [groups[int(i)] for i in sampled if groups[int(i)].size]
    if not parts:
        return np.zeros(0, dtype=int)
    return np.concatenate(parts, axis=0)


def bootstrap_metric(
    pack: HeatmapPack,
    fn: Callable[[HeatmapPack], float],
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    _clips, _inv, groups = _clip_groups(pack.clip_id)
    rng = np.random.RandomState(seed)
    n_clips = len(groups)
    stats = []
    for _ in range(int(n_boot)):
        samp = rng.randint(0, n_clips, size=n_clips)
        idx = _resample_from_groups(groups, samp)
        if idx.size == 0:
            continue
        val = fn(pack.take(idx))
        if np.isfinite(val):
            stats.append(float(val))
    if not stats:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    arr = np.asarray(stats, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "lo": float(np.quantile(arr, 0.025)),
        "hi": float(np.quantile(arr, 0.975)),
    }


def bootstrap_p_better(
    pack_a: HeatmapPack,
    pack_b: HeatmapPack,
    fn: Callable[[HeatmapPack], float],
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """P(metric(A) > metric(B)) under the same clip bootstrap."""
    clips_a, _inv_a, groups_a = _clip_groups(pack_a.clip_id)
    clips_b, _inv_b, groups_b = _clip_groups(pack_b.clip_id)
    if not np.array_equal(clips_a, clips_b):
        common = np.intersect1d(clips_a, clips_b)
        map_a = {c: i for i, c in enumerate(clips_a)}
        map_b = {c: i for i, c in enumerate(clips_b)}
        groups_a = [groups_a[map_a[c]] for c in common]
        groups_b = [groups_b[map_b[c]] for c in common]
    n_clips = len(groups_a)
    rng = np.random.RandomState(seed)
    wins = 0
    n = 0
    for _ in range(int(n_boot)):
        samp = rng.randint(0, n_clips, size=n_clips)
        ia = _resample_from_groups(groups_a, samp)
        ib = _resample_from_groups(groups_b, samp)
        if ia.size == 0 or ib.size == 0:
            continue
        va = fn(pack_a.take(ia))
        vb = fn(pack_b.take(ib))
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        n += 1
        wins += int(va > vb)
    return float(wins / n) if n else float("nan")


def compare_models(
    packs: dict[str, HeatmapPack],
    *,
    prior_name: str = "mean_prior",
    warning: dict | None = None,
    seed: int = 0,
    n_folds: int = 5,
    n_boot: int = 200,
    p_better_min: float = 0.90,
) -> dict[str, Any]:
    """Score every pack, gate vs the mean prior, Borda-rank eligible models."""
    scoreboard = {name: score_pack(pack, warning=warning, seed=seed, n_folds=n_folds) for name, pack in packs.items()}
    if prior_name not in scoreboard:
        raise KeyError(f"prior pack '{prior_name}' missing")
    prior = scoreboard[prior_name]
    gate = {name: gates_vs_prior(m, prior) for name, m in scoreboard.items()}
    gate[prior_name] = {
        "pass": False,
        "beats_prior_clip_ap": False,
        "beats_prior_weighted_mse": False,
        "not_spatially_collapsed": True,
        "is_prior": True,
    }
    eligible = [n for n in packs if n != prior_name and gate[n]["pass"]]
    ranks_all = borda_ranks(scoreboard, list(packs.keys()))
    ranks_elig = borda_ranks(scoreboard, eligible) if eligible else {}
    winner = None
    unique = False
    reason = "no model beat the mean-heatmap prior on clip AP and weighted MSE"
    pairwise: dict[str, Any] = {}
    if eligible:
        winner = min(eligible, key=lambda n: (ranks_elig[n], n))
        others = [n for n in eligible if n != winner]
        reason = "lowest Borda rank among models that beat the mean heatmap"
        if not others:
            unique = True
            reason = "only eligible model"
        else:
            runner = min(others, key=lambda n: (ranks_elig[n], n))
            def _clip_ap(p: HeatmapPack) -> float:
                return float(clip_aps(p)["clip_macro_ap"])

            def _peak(p: HeatmapPack) -> float:
                return float(peak_metrics(p)["peak_hit"])

            def _rec5(p: HeatmapPack) -> float:
                return float(nested_recall_at_fa(p, fa_rate=0.05, n_folds=n_folds, seed=seed)["recall"])

            p_ap = bootstrap_p_better(packs[winner], packs[runner], _clip_ap, n_boot=n_boot, seed=seed)
            p_peak = bootstrap_p_better(packs[winner], packs[runner], _peak, n_boot=n_boot, seed=seed)
            p_rec = bootstrap_p_better(packs[winner], packs[runner], _rec5, n_boot=n_boot, seed=seed)
            pairwise = {
                "runner_up": runner,
                "p_better_clip_macro_ap": p_ap,
                "p_better_peak_hit": p_peak,
                "p_better_nested_rec_5fa": p_rec,
            }
            n_sig = int(p_ap >= p_better_min) + int(p_peak >= p_better_min) + int(p_rec >= p_better_min)
            unique = bool(n_sig >= 2 and ranks_elig[runner] - ranks_elig[winner] >= 0.15)
            if not unique:
                reason = (
                    f"Borda prefers {winner} over {runner}, but clip-bootstrap "
                    f"does not separate them on at least two primary metrics "
                    f"(p_better AP={p_ap:.2f} peak={p_peak:.2f} rec5={p_rec:.2f})"
                )
            else:
                reason = f"Borda winner {winner}; bootstrap-separated from {runner}"
    return {
        "metrics": scoreboard,
        "gates": gate,
        "eligible": eligible,
        "borda_all": ranks_all,
        "borda_eligible": ranks_elig,
        "winner": winner,
        "unique": unique,
        "reason": reason,
        "pairwise": pairwise,
        "rank_metrics": [k for k, _ in RANK_METRICS],
    }
