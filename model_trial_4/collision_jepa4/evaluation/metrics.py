"""Per-horizon heatmap and warning metrics. Separate from training."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SAFE, CAUTION, STOP = 0, 1, 2


def direction_scores_np(heatmap: np.ndarray, row_weights: list[float], left, center, right) -> dict[str, float]:
    hm = np.asarray(heatmap, dtype=np.float32)
    w = np.asarray(row_weights, dtype=np.float32).reshape(-1, 1)
    weighted = hm * w
    return {
        "LEFT": float(weighted[:, left].max()) if left else 0.0,
        "CENTER": float(weighted[:, center].max()) if center else 0.0,
        "RIGHT": float(weighted[:, right].max()) if right else 0.0,
    }


def classify_heatmap(heatmap: np.ndarray, cfg_warning: dict) -> tuple[str, int, dict[str, float]]:
    """Direction from unweighted side maxima. Severity from the plain map max.

    Warning row weights are not applied: a far-row occupied cell can reach STOP.
    """
    ones = [1.0] * int(np.asarray(heatmap).shape[0])
    scores = direction_scores_np(
        heatmap,
        ones,
        cfg_warning["left_cols"],
        cfg_warning["center_cols"],
        cfg_warning["right_cols"],
    )
    direction = max(scores, key=scores.get)
    peak = float(np.max(np.asarray(heatmap, dtype=np.float32)))
    if peak >= float(cfg_warning["stop_threshold"]):
        sev = STOP
    elif peak >= float(cfg_warning["caution_threshold"]):
        sev = CAUTION
    else:
        sev = SAFE
    return direction, sev, scores


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


@dataclass
class HorizonAccumulator:
    name: str
    high_threat_threshold: float = 0.5
    safe_cell_threshold: float = 0.05
    warning: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n = 0
        self.masked = 0
        self.sq = 0.0
        self.wsq = 0.0
        self.wn = 0.0
        self.cells = 0
        self.tp = self.fp = self.fn = 0
        self.safe_cells = 0
        self.safe_fp_cells = 0
        self.dir_correct = 0
        self.dir_total = 0
        self.stop_tp = self.stop_fp = self.stop_fn = 0
        self.n_safe_frames = 0
        self.n_stop_frames = 0
        self.nuisance_n = 0
        self.miss_n = 0
        self.safe_stop_n = 0
        self.by_outcome: dict[str, list[float]] = {}
        self.by_background: dict[str, list[float]] = {}
        self.by_scenario: dict[str, list[float]] = {}
        self.pre_fork_n = 0
        self.post_fork_n = 0
        self.pre_fork_mse = 0.0
        self.post_fork_mse = 0.0
        self.pre_fork_stop_tp = 0
        self.pre_fork_stop_n = 0
        self.post_fork_stop_tp = 0
        self.post_fork_stop_n = 0
        self.post_fork_safe_n = 0
        self.post_fork_safe_stop_n = 0

    def update(
        self,
        pred: np.ndarray,
        true: np.ndarray,
        *,
        masked: bool,
        outcome: str,
        background: str,
        scenario: str,
        pre_fork: bool,
    ) -> None:
        if masked:
            self.masked += 1
            return
        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(true, dtype=np.float32)
        self.n += 1
        se = (pred - true) ** 2
        self.sq += float(se.sum())
        self.cells += int(pred.size)
        w = 0.05 + 4.0 * np.square(true)
        w = w / max(float(w.mean()), 1e-6)
        self.wsq += float((se * w).sum())
        self.wn += float(w.sum())
        ht = self.high_threat_threshold
        p_high = pred >= ht
        t_high = true >= ht
        self.tp += int(np.sum(p_high & t_high))
        self.fp += int(np.sum(p_high & ~t_high))
        self.fn += int(np.sum(~p_high & t_high))
        safe = true < self.safe_cell_threshold
        self.safe_cells += int(np.sum(safe))
        self.safe_fp_cells += int(np.sum(safe & (pred >= ht)))
        p_dir, p_sev, _ = classify_heatmap(pred, self.warning)
        t_dir, t_sev, _ = classify_heatmap(true, self.warning)
        if t_sev >= CAUTION:
            self.dir_total += 1
            self.dir_correct += int(p_dir == t_dir)
        p_stop = p_sev >= STOP
        t_stop = t_sev >= STOP
        self.stop_tp += int(p_stop and t_stop)
        self.stop_fp += int(p_stop and not t_stop)
        self.stop_fn += int((not p_stop) and t_stop)
        if t_sev == SAFE:
            self.n_safe_frames += 1
            self.nuisance_n += int(p_sev >= CAUTION)
            self.safe_stop_n += int(p_sev >= STOP)
        if t_sev >= STOP:
            self.n_stop_frames += 1
            self.miss_n += int(p_sev == SAFE)
        mse = float(se.mean())
        self.by_outcome.setdefault(outcome, []).append(mse)
        self.by_background.setdefault(background, []).append(mse)
        self.by_scenario.setdefault(scenario, []).append(mse)
        if pre_fork:
            self.pre_fork_n += 1
            self.pre_fork_mse += mse
            if t_stop:
                self.pre_fork_stop_n += 1
                self.pre_fork_stop_tp += int(p_stop)
        else:
            self.post_fork_n += 1
            self.post_fork_mse += mse
            if t_stop:
                self.post_fork_stop_n += 1
                self.post_fork_stop_tp += int(p_stop)
            if t_sev == SAFE:
                self.post_fork_safe_n += 1
                self.post_fork_safe_stop_n += int(p_stop)

    def summary(self) -> dict[str, float]:
        mse = self.sq / self.cells if self.cells else float("nan")
        wmse = self.wsq / self.wn if self.wn else float("nan")
        ht_p, ht_r, ht_f1 = _prf(self.tp, self.fp, self.fn)
        stop_p, stop_r, stop_f1 = _prf(self.stop_tp, self.stop_fp, self.stop_fn)
        fp_rate = self.safe_fp_cells / self.safe_cells if self.safe_cells else 0.0
        dir_acc = self.dir_correct / self.dir_total if self.dir_total else 0.0
        nuisance = self.nuisance_n / self.n_safe_frames if self.n_safe_frames else 0.0
        miss = self.miss_n / self.n_stop_frames if self.n_stop_frames else 0.0
        total_seen = self.n + self.masked
        return {
            "n": float(self.n),
            "masked": float(self.masked),
            "masked_frac": self.masked / total_seen if total_seen else 0.0,
            "mse": mse,
            "weighted_mse": wmse,
            "high_threat_precision": ht_p,
            "high_threat_recall": ht_r,
            "high_threat_f1": ht_f1,
            "false_positive_rate_safe_cells": fp_rate,
            "direction_accuracy": dir_acc,
            "stop_precision": stop_p,
            "stop_recall": stop_r,
            "stop_f1": stop_f1,
            "nuisance": nuisance,
            "miss": miss,
            "safe_recall": (1.0 - nuisance) if self.n_safe_frames else float("nan"),
            "safe_stop_rate": (
                self.safe_stop_n / self.n_safe_frames if self.n_safe_frames else float("nan")
            ),
            "pre_fork_mse": self.pre_fork_mse / self.pre_fork_n if self.pre_fork_n else float("nan"),
            "post_fork_mse": self.post_fork_mse / self.post_fork_n if self.post_fork_n else float("nan"),
            "pre_fork_n": float(self.pre_fork_n),
            "post_fork_n": float(self.post_fork_n),
            "pre_fork_stop_recall": (
                self.pre_fork_stop_tp / self.pre_fork_stop_n if self.pre_fork_stop_n else float("nan")
            ),
            "post_fork_stop_recall": (
                self.post_fork_stop_tp / self.post_fork_stop_n if self.post_fork_stop_n else float("nan")
            ),
            "pre_fork_stop_n": float(self.pre_fork_stop_n),
            "post_fork_stop_n": float(self.post_fork_stop_n),
            "post_fork_safe_stop_rate": (
                self.post_fork_safe_stop_n / self.post_fork_safe_n if self.post_fork_safe_n else float("nan")
            ),
        }
