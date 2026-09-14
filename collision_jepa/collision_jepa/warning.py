"""Turn a predicted 5x5 heatmap into a stabilized wearable warning.

Pipeline (no neural network):
  5x5 heatmap  ->  LEFT / CENTER / RIGHT direction scores (near-weighted)
               ->  SAFE / CAUTION / STOP severity
               ->  HitLatch hysteresis (anti-flicker)

Everything here is plain numpy so it can run on-device and inside metrics without
a torch dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SAFE, CAUTION, STOP = 0, 1, 2
_SEVERITY_NAMES = {SAFE: "SAFE", CAUTION: "CAUTION", STOP: "STOP"}


def severity_name(sev: int) -> str:
    return _SEVERITY_NAMES.get(int(sev), "SAFE")


def direction_scores(
    heatmap: np.ndarray,
    row_weights: list[float],
    left_cols: list[int],
    center_cols: list[int],
    right_cols: list[int],
) -> dict[str, float]:
    """Near-weighted threat score per direction.

    Each cell is weighted by its row weight (near rows matter more), then each
    direction takes the max weighted value over its columns. A strong threat close
    to the person dominates a weak one far away.
    """
    hm = np.asarray(heatmap, dtype=np.float32)
    w = np.asarray(row_weights, dtype=np.float32).reshape(-1, 1)
    weighted = hm * w  # [grid, grid]
    groups = {"LEFT": left_cols, "CENTER": center_cols, "RIGHT": right_cols}
    scores: dict[str, float] = {}
    for name, cols in groups.items():
        if cols:
            scores[name] = float(weighted[:, cols].max())
        else:
            scores[name] = 0.0
    return scores


def severity_from_score(score: float, caution_threshold: float, stop_threshold: float) -> int:
    if score >= stop_threshold:
        return STOP
    if score >= caution_threshold:
        return CAUTION
    return SAFE


def classify(heatmap: np.ndarray, cfg_warning: dict) -> tuple[str, int, dict[str, float]]:
    """Return (direction, severity, scores) for a single heatmap."""
    scores = direction_scores(
        heatmap,
        cfg_warning["row_weights"],
        cfg_warning["left_cols"],
        cfg_warning["center_cols"],
        cfg_warning["right_cols"],
    )
    direction = max(scores, key=scores.get)
    severity = severity_from_score(
        scores[direction],
        cfg_warning["caution_threshold"],
        cfg_warning["stop_threshold"],
    )
    return direction, severity, scores


@dataclass
class HitLatch:
    """Temporal hysteresis so a single frame cannot flip the warning.

    A danger (severity >= CAUTION) must persist for ``confirm_frames`` before the
    warning turns on; once on, it stays on until ``release_frames`` consecutive SAFE
    frames. This is the same anti-flicker idea as the original YOLO demo's HitLatch.
    """

    confirm_frames: int = 3
    release_frames: int = 5

    def __post_init__(self) -> None:
        self.active: bool = False
        self.stable_severity: int = SAFE
        self.stable_direction: str = "CENTER"
        self._danger_streak: int = 0
        self._safe_streak: int = 0
        self._pending_severity: int = SAFE
        self._pending_direction: str = "CENTER"

    def update(self, raw_severity: int, raw_direction: str) -> tuple[str, int]:
        """Feed one frame's raw prediction, get the stabilized (direction, severity)."""
        if raw_severity >= CAUTION:
            self._danger_streak += 1
            self._safe_streak = 0
            self._pending_severity = raw_severity
            self._pending_direction = raw_direction
        else:
            self._safe_streak += 1
            self._danger_streak = 0

        if not self.active:
            if self._danger_streak >= self.confirm_frames:
                self.active = True
                self.stable_severity = self._pending_severity
                self.stable_direction = self._pending_direction
        else:
            if self._safe_streak >= self.release_frames:
                self.active = False
                self.stable_severity = SAFE
                self.stable_direction = raw_direction
            elif raw_severity >= CAUTION:
                # Stay on; track the latest (possibly escalated) danger.
                self.stable_severity = raw_severity
                self.stable_direction = raw_direction

        return self.stable_direction, self.stable_severity

    def reset(self) -> None:
        self.__post_init__()
