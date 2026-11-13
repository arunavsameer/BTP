"""Event labels, threat side, and time-to-impact from clip metadata + 5×5 GT maps."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import classify_heatmap

FPS_DEFAULT = 30.0
HIT_HORIZONS_S = (1.0, 2.0, 3.0, 4.0)
NEAR_ROWS = (3, 4)  # closest two rows of the 5×5 grid
DEFAULT_WARNING = {
    "row_weights": [1.0, 1.0, 1.0, 1.0, 1.0],
    "left_cols": [0, 1],
    "center_cols": [2],
    "right_cols": [3, 4],
    "caution_threshold": 0.25,
    "stop_threshold": 0.60,
}


def clip_outcome_class(outcome_type: str) -> str:
    """collision / miss / safe. This dataset has no dedicated safe clips."""
    o = str(outcome_type)
    if o == "collision":
        return "collision"
    if o.startswith("miss"):
        return "miss"
    if o == "safe":
        return "safe"
    return o


def miss_side(outcome_type: str) -> str | None:
    o = str(outcome_type)
    if o == "miss_left":
        return "LEFT"
    if o == "miss_right":
        return "RIGHT"
    return None


def collision_frame(outcome_type: str, hit_s: float | None, fps: float) -> int | None:
    if clip_outcome_class(outcome_type) != "collision":
        return None
    if hit_s is None:
        return None
    return int(round(float(hit_s) * float(fps)))


def time_to_impact_s(frame: int, collision_f: int | None, fps: float) -> float | None:
    if collision_f is None:
        return None
    return (int(collision_f) - int(frame)) / float(fps)


def collision_within(tti_s: float | None, horizon_s: float) -> bool:
    if tti_s is None:
        return False
    return 0.0 < float(tti_s) <= float(horizon_s)


def side_from_scenario(scenario_type: str) -> str | None:
    n = str(scenario_type).lower()
    if "from_left" in n:
        return "LEFT"
    if "from_right" in n:
        return "RIGHT"
    if "head_on" in n or "on_path" in n:
        return "CENTER"
    return None


def side_from_heatmap(heatmap: np.ndarray, warning: dict | None = None) -> str:
    warn = warning or DEFAULT_WARNING
    direction, _sev, _ = classify_heatmap(heatmap, warn)
    return str(direction)


def heatmap_near_row_max(heatmap: np.ndarray, warning: dict | None = None) -> float:
    """Peak used for STOP/CAUTION: plain max of the 5×5 (row weights ignored)."""
    return float(np.max(np.asarray(heatmap, dtype=np.float64)))


def heatmap_alert(
    maps: dict[str, np.ndarray],
    *,
    cutoff: float,
    warning: dict | None = None,
    amb: dict[str, bool] | None = None,
) -> tuple[bool, str, float]:
    """Alert if any unmasked horizon's plain max is above cutoff. Side from the loudest map."""
    best_peak = -1.0
    best_side = "CENTER"
    fired = False
    for name, hm in maps.items():
        if amb and amb.get(name, False):
            continue
        peak = heatmap_near_row_max(hm, warning)
        if peak > best_peak:
            best_peak = peak
            best_side = side_from_heatmap(hm, warning)
        if peak >= float(cutoff):
            fired = True
    return fired, best_side, best_peak


def event_alert(tti_s: float | None, horizon_s: float = 4.0) -> bool:
    return collision_within(tti_s, horizon_s)


def true_side_for_group(
    collision_heatmaps: np.ndarray,
    scenario_type: str,
    warning: dict | None = None,
) -> str:
    """Prefer a named side in the scenario; otherwise the collision clip's peak heatmap."""
    named = side_from_scenario(scenario_type)
    if named is not None:
        return named
    peaks = np.asarray(collision_heatmaps, dtype=np.float64).reshape(len(collision_heatmaps), -1).max(axis=1)
    idx = int(np.argmax(peaks))
    return side_from_heatmap(collision_heatmaps[idx], warning)


def row_scalars(
    *,
    clip_id: str,
    sample_id: str,
    frame: int,
    fps: float,
    outcome_type: str,
    fork_frame: int,
    hit_s: float | None,
    scenario_type: str,
    true_side: str,
    background_id: str,
    background_pair_id: str,
    outcome_group_id: str,
    amb: dict[str, bool],
) -> dict[str, Any]:
    col_f = collision_frame(outcome_type, hit_s, fps)
    tti = time_to_impact_s(frame, col_f, fps)
    oc = clip_outcome_class(outcome_type)
    within = {f"collision_within_{int(h)}s": bool(collision_within(tti, h)) for h in HIT_HORIZONS_S}
    if oc == "collision" and tti is not None and tti <= 0:
        anchor_class = "post_impact"
    elif within["collision_within_4s"]:
        anchor_class = "collision"
    elif oc == "miss":
        anchor_class = "miss"
    else:
        anchor_class = "safe"
    return {
        "clip_id": clip_id,
        "sample_id": sample_id,
        "frame": int(frame),
        "fps": float(fps),
        "outcome": oc,
        "outcome_raw": str(outcome_type),
        "miss_side": miss_side(outcome_type),
        "fork_frame": int(fork_frame),
        "collision_frame": col_f,
        "hit_s": None if hit_s is None else float(hit_s),
        "scenario_type": str(scenario_type),
        "true_side": str(true_side),
        "background_id": str(background_id),
        "background_pair_id": str(background_pair_id),
        "outcome_group_id": str(outcome_group_id),
        "pre_fork": bool(int(frame) < int(fork_frame)),
        "amb_1s": bool(amb.get("1s", False)),
        "amb_1_5s": bool(amb.get("1_5s", False)),
        "amb_2s": bool(amb.get("2s", False)),
        "tti_s": tti,
        "anchor_class": anchor_class,
        **within,
    }
