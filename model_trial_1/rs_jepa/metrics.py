"""Evaluation metrics for the k×k collision heatmap and wearable beep."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .warning import CAUTION, SAFE, STOP, HitLatch, classify


@dataclass
class MetricAccumulator:
    recall_threshold: float = 0.5
    cfg_warning: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sq_err_sum = 0.0
        self._n_cells = 0
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._dir_correct = 0
        self._dir_total = 0
        self._stop_tp = 0
        self._stop_fp = 0
        self._stop_fn = 0
        self._n = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(true, dtype=np.float32)
        self._n += 1
        self._sq_err_sum += float(np.sum((pred - true) ** 2))
        self._n_cells += pred.size

        thr = self.recall_threshold
        p_high = pred >= thr
        t_high = true >= thr
        self._tp += int(np.sum(p_high & t_high))
        self._fp += int(np.sum(p_high & ~t_high))
        self._fn += int(np.sum(~p_high & t_high))

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


def selection_cfg(cfg) -> dict:
    get = cfg.get if hasattr(cfg, "get") else lambda k, d=None: (cfg or {}).get(k, d)
    return {
        "nuisance_weight": float(get("selection.nuisance_weight", 0.5)),
        "miss_weight": float(get("selection.miss_weight", 0.5)),
        "max_nuisance": float(get("selection.max_nuisance", 0.20)),
        "max_miss": float(get("selection.max_miss", 0.25)),
        "use_latch": bool(get("selection.use_latch", True)),
    }


@dataclass
class WearableAccumulator:
    """Latched beep errors: nuisance = P(beep|SAFE), miss = P(silent|STOP)."""

    cfg_warning: dict = field(default_factory=dict)
    use_latch: bool = True
    nuisance_weight: float = 0.5
    miss_weight: float = 0.5
    max_nuisance: float = 0.20
    max_miss: float = 0.25

    def __post_init__(self) -> None:
        self._n_safe = 0
        self._n_stop = 0
        self._nuisance_n = 0
        self._miss_n = 0
        self._n = 0
        self._latch = self._new_latch()

    def _new_latch(self) -> HitLatch | None:
        if not self.use_latch:
            return None
        return HitLatch(
            int(self.cfg_warning.get("confirm_frames", 3)),
            int(self.cfg_warning.get("release_frames", 5)),
        )

    def start_episode(self) -> None:
        self._latch = self._new_latch()

    def update_severities(self, pred_severity: int, true_severity: int) -> None:
        self._n += 1
        beep = int(pred_severity) >= CAUTION
        silent = int(pred_severity) == SAFE
        if int(true_severity) == SAFE:
            self._n_safe += 1
            self._nuisance_n += int(beep)
        if int(true_severity) >= STOP:
            self._n_stop += 1
            self._miss_n += int(silent)

    def update_heatmaps(self, pred: np.ndarray, true: np.ndarray) -> None:
        p_dir, p_raw, _ = classify(pred, self.cfg_warning)
        _, t_sev, _ = classify(true, self.cfg_warning)
        if self._latch is not None:
            _, p_sev = self._latch.update(p_raw, p_dir)
        else:
            p_sev = p_raw
        self.update_severities(p_sev, t_sev)

    def summary(self) -> dict[str, float]:
        nuisance = self._nuisance_n / self._n_safe if self._n_safe else 0.0
        miss = self._miss_n / self._n_stop if self._n_stop else 0.0
        score = 1.0 - self.nuisance_weight * nuisance - self.miss_weight * miss
        eligible = (nuisance <= self.max_nuisance) and (miss <= self.max_miss)
        return {
            "n": self._n,
            "n_safe": self._n_safe,
            "n_stop": self._n_stop,
            "nuisance": nuisance,
            "miss": miss,
            "score": score,
            "eligible": float(eligible),
            "max_nuisance": self.max_nuisance,
            "max_miss": self.max_miss,
        }


def format_wearable(name: str, summary: dict[str, float]) -> str:
    gate = "eligible" if summary.get("eligible") else "ineligible"
    return (
        f"{name:<16} "
        f"score={summary['score']:.3f}  "
        f"nuisance={summary['nuisance']:.3f}  "
        f"miss={summary['miss']:.3f}  "
        f"{gate}"
    )


def wearable_from_cfg(cfg) -> WearableAccumulator:
    sel = selection_cfg(cfg)
    warn = cfg.get("warning") if hasattr(cfg, "get") else cfg["warning"]
    return WearableAccumulator(
        cfg_warning=warn,
        use_latch=bool(sel["use_latch"]),
        nuisance_weight=sel["nuisance_weight"],
        miss_weight=sel["miss_weight"],
        max_nuisance=sel["max_nuisance"],
        max_miss=sel["max_miss"],
    )
