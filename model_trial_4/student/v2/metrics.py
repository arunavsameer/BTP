"""Extra V2 eval: center-column STOP vs teacher."""

from __future__ import annotations

import numpy as np


def center_col_stop_recall(pred: np.ndarray, teacher: np.ndarray, col: int = 2, thr: float = 0.60) -> float:
    """pred/teacher: [N, 5, 5]. Recall of teacher-col STOP on student col."""
    t_peak = teacher[:, :, int(col)].max(axis=1)
    p_peak = pred[:, :, int(col)].max(axis=1)
    need = t_peak >= float(thr)
    if not np.any(need):
        return float("nan")
    return float(np.mean(p_peak[need] >= float(thr)))
