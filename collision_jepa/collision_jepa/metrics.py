"""Evaluation metrics for the 5x5 collision heatmap and the warning it produces.

We deliberately do NOT rely on MSE alone: the center-corridor prior makes MSE look
good for a model that ignores real hazards. The metrics that matter for a blind user
are high-threat recall and STOP precision/recall/F1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .warning import CAUTION, STOP, classify


@dataclass
class MetricAccumulator:
    """Accumulate predictions over a dataset, then summarize.

    All heatmaps are numpy arrays of shape [grid, grid]. ``cfg_warning`` is the
    warning sub-config (thresholds, row weights, column groups).
    """

    recall_threshold: float = 0.5
    cfg_warning: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sq_err_sum = 0.0
        self._n_cells = 0
        # High-threat cell detection (pred high vs true high).
        self._tp = 0
        self._fp = 0
        self._fn = 0
        # Direction agreement (only on frames where the truth is not SAFE).
        self._dir_correct = 0
        self._dir_total = 0
        # STOP confusion (frame-level, no hysteresis).
        self._stop_tp = 0
        self._stop_fp = 0
        self._stop_fn = 0
        self._n = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(true, dtype=np.float32)
        self._n += 1

        # MSE
        self._sq_err_sum += float(np.sum((pred - true) ** 2))
        self._n_cells += pred.size

        # High-threat cell detection
        thr = self.recall_threshold
        p_high = pred >= thr
        t_high = true >= thr
        self._tp += int(np.sum(p_high & t_high))
        self._fp += int(np.sum(p_high & ~t_high))
        self._fn += int(np.sum(~p_high & t_high))

        # Direction + STOP via the warning readout.
        p_dir, p_sev, _ = classify(pred, self.cfg_warning)
        t_dir, t_sev, _ = classify(true, self.cfg_warning)

        if t_sev >= CAUTION:
            self._dir_total += 1
            if p_dir == t_dir:
                self._dir_correct += 1

        p_stop = p_sev >= STOP
        t_stop = t_sev >= STOP
        self._stop_tp += int(p_stop and t_stop)
        self._stop_fp += int(p_stop and not t_stop)
        self._stop_fn += int(not p_stop and t_stop)

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    def summary(self) -> dict[str, float]:
        mse = self._sq_err_sum / self._n_cells if self._n_cells else 0.0
        ht_p, ht_r, ht_f1 = self._prf(self._tp, self._fp, self._fn)
        stop_p, stop_r, stop_f1 = self._prf(self._stop_tp, self._stop_fp, self._stop_fn)
        dir_acc = self._dir_correct / self._dir_total if self._dir_total else 0.0
        return {
            "n": self._n,
            "mse": mse,
            "high_threat_precision": ht_p,
            "high_threat_recall": ht_r,
            "high_threat_f1": ht_f1,
            "direction_accuracy": dir_acc,
            "stop_precision": stop_p,
            "stop_recall": stop_r,
            "stop_f1": stop_f1,
        }


def format_summary(name: str, summary: dict[str, float]) -> str:
    return (
        f"{name:<16} "
        f"MSE={summary['mse']:.4f}  "
        f"HT-recall={summary['high_threat_recall']:.3f}  "
        f"HT-F1={summary['high_threat_f1']:.3f}  "
        f"dir-acc={summary['direction_accuracy']:.3f}  "
        f"STOP-F1={summary['stop_f1']:.3f} "
        f"(P={summary['stop_precision']:.3f} R={summary['stop_recall']:.3f})"
    )
