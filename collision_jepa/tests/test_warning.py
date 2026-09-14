"""Tests for the warning readout and HitLatch hysteresis.

Run:  python tests/test_warning.py   (or: pytest tests/test_warning.py)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.warning import (  # noqa: E402
    CAUTION,
    HitLatch,
    SAFE,
    STOP,
    classify,
    direction_scores,
    severity_from_score,
)

WARN_CFG = {
    "left_cols": [0, 1],
    "center_cols": [2],
    "right_cols": [3, 4],
    "row_weights": [0.4, 0.6, 0.8, 1.0, 1.0],
    "caution_threshold": 0.45,
    "stop_threshold": 0.75,
    "confirm_frames": 3,
    "release_frames": 5,
}


def test_direction_center_hazard():
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[3, 2] = 0.9  # near center
    d, sev, scores = classify(hm, WARN_CFG)
    assert d == "CENTER", (d, scores)
    assert sev == STOP


def test_direction_left_hazard():
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[3, 0] = 0.8
    d, sev, _ = classify(hm, WARN_CFG)
    assert d == "LEFT"
    assert sev == STOP


def test_near_weighting_beats_far():
    # A far strong cell vs a near moderate cell; near should win after weighting.
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[0, 0] = 0.6   # far-left, weight 0.4 -> 0.24
    hm[4, 4] = 0.6   # near-right, weight 1.0 -> 0.60
    scores = direction_scores(hm, WARN_CFG["row_weights"], [0, 1], [2], [3, 4])
    assert scores["RIGHT"] > scores["LEFT"]


def test_severity_thresholds():
    assert severity_from_score(0.2, 0.45, 0.75) == SAFE
    assert severity_from_score(0.5, 0.45, 0.75) == CAUTION
    assert severity_from_score(0.9, 0.45, 0.75) == STOP


def test_hitlatch_requires_confirm():
    latch = HitLatch(confirm_frames=3, release_frames=5)
    # Two danger frames: not yet active.
    latch.update(STOP, "CENTER")
    _, sev = latch.update(STOP, "CENTER")
    assert sev == SAFE and not latch.active
    # Third danger frame: turns on.
    _, sev = latch.update(STOP, "CENTER")
    assert sev == STOP and latch.active


def test_hitlatch_release_needs_persistence():
    latch = HitLatch(confirm_frames=2, release_frames=3)
    latch.update(STOP, "CENTER")
    _, sev = latch.update(STOP, "CENTER")
    assert latch.active and sev == STOP
    # A couple of safe frames should NOT immediately release.
    latch.update(SAFE, "CENTER")
    _, sev = latch.update(SAFE, "CENTER")
    assert latch.active and sev == STOP
    # Third consecutive safe frame releases.
    _, sev = latch.update(SAFE, "CENTER")
    assert not latch.active and sev == SAFE


def test_hitlatch_safe_streak_resets_on_danger():
    latch = HitLatch(confirm_frames=2, release_frames=3)
    latch.update(STOP, "CENTER")
    latch.update(STOP, "CENTER")
    latch.update(SAFE, "CENTER")
    latch.update(SAFE, "CENTER")
    # Danger interrupts the release streak.
    latch.update(STOP, "CENTER")
    _, sev = latch.update(SAFE, "CENTER")
    assert latch.active and sev in (STOP, CAUTION)


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} warning tests passed")


if __name__ == "__main__":
    _run()
