"""Wearable nuisance / miss selection."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.engine import wearable_better  # noqa: E402
from rs_jepa.metrics import WearableAccumulator  # noqa: E402
from rs_jepa.warning import CAUTION, SAFE, STOP  # noqa: E402


def test_perfect_balance() -> None:
    w = WearableAccumulator(use_latch=False)
    for _ in range(8):
        w.update_severities(SAFE, SAFE)
    for _ in range(8):
        w.update_severities(STOP, STOP)
    s = w.summary()
    assert s["nuisance"] == 0.0
    assert s["miss"] == 0.0
    assert abs(s["score"] - 1.0) < 1e-6
    assert s["eligible"] == 1.0


def test_false_beeps_fail_gate() -> None:
    w = WearableAccumulator(use_latch=False, max_nuisance=0.20)
    for _ in range(10):
        w.update_severities(STOP, SAFE)
    s = w.summary()
    assert s["nuisance"] == 1.0
    assert s["miss"] == 0.0
    assert s["eligible"] == 0.0
    assert abs(s["score"] - 0.5) < 1e-6


def test_mute_on_stop_fails_gate() -> None:
    w = WearableAccumulator(use_latch=False, max_miss=0.25)
    for _ in range(10):
        w.update_severities(SAFE, STOP)
    s = w.summary()
    assert s["miss"] == 1.0
    assert s["eligible"] == 0.0


def test_caution_on_safe_is_nuisance() -> None:
    w = WearableAccumulator(use_latch=False)
    w.update_severities(CAUTION, SAFE)
    w.update_severities(SAFE, SAFE)
    s = w.summary()
    assert abs(s["nuisance"] - 0.5) < 1e-6


def test_caution_on_stop_is_not_a_miss() -> None:
    w = WearableAccumulator(use_latch=False)
    w.update_severities(CAUTION, STOP)
    s = w.summary()
    assert s["miss"] == 0.0


def test_prefer_eligible_over_higher_ineligible_score() -> None:
    hot = {"score": 0.95, "eligible": 0.0}
    ok = {"score": 0.80, "eligible": 1.0}
    assert wearable_better(ok, hot)
    assert not wearable_better(hot, ok)


def test_latch_needs_confirm() -> None:
    warn = {
        "row_weights": [1.0, 1.0, 1.0],
        "left_cols": [0],
        "center_cols": [1],
        "right_cols": [2],
        "caution_threshold": 0.42,
        "stop_threshold": 0.70,
        "confirm_frames": 3,
        "release_frames": 5,
    }
    w = WearableAccumulator(cfg_warning=warn, use_latch=True)
    # two STOP frames: latch not yet active → still silent (not a miss after 2)
    stop_hm = __import__("numpy").zeros((3, 3), dtype="float32")
    stop_hm[2, 1] = 1.0
    safe_hm = __import__("numpy").zeros((3, 3), dtype="float32")
    w.start_episode()
    w.update_heatmaps(stop_hm, stop_hm)
    w.update_heatmaps(stop_hm, stop_hm)
    s = w.summary()
    assert s["miss"] == 1.0  # still silent until confirm=3
    w.update_heatmaps(stop_hm, stop_hm)
    s = w.summary()
    assert abs(s["miss"] - 2.0 / 3.0) < 1e-6  # third frame latches, not a miss


if __name__ == "__main__":
    test_perfect_balance()
    test_false_beeps_fail_gate()
    test_mute_on_stop_fails_gate()
    test_caution_on_safe_is_nuisance()
    test_caution_on_stop_is_not_a_miss()
    test_prefer_eligible_over_higher_ineligible_score()
    test_latch_needs_confirm()
    print("ok")
