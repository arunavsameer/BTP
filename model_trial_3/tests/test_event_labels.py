from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.evaluation.labels import (
    clip_outcome_class,
    collision_frame,
    collision_within,
    event_alert,
    heatmap_alert,
    side_from_scenario,
    time_to_impact_s,
)


class EventLabelTests(unittest.TestCase):
    def test_outcomes(self) -> None:
        self.assertEqual(clip_outcome_class("collision"), "collision")
        self.assertEqual(clip_outcome_class("miss_left"), "miss")
        self.assertEqual(clip_outcome_class("miss_right"), "miss")

    def test_tti_and_windows(self) -> None:
        col = collision_frame("collision", 14.35, 30.0)
        self.assertEqual(col, 430)
        self.assertIsNone(collision_frame("miss_left", 14.35, 30.0))
        tti = time_to_impact_s(310, col, 30.0)
        self.assertAlmostEqual(tti, 4.0, places=5)
        self.assertTrue(collision_within(tti, 4.0))
        self.assertFalse(collision_within(tti, 3.0))
        self.assertFalse(collision_within(None, 4.0))
        self.assertFalse(event_alert(-0.1, 4.0))

    def test_scenario_side(self) -> None:
        self.assertEqual(side_from_scenario("cube_from_left"), "LEFT")
        self.assertEqual(side_from_scenario("jaywalker_from_right"), "RIGHT")
        self.assertEqual(side_from_scenario("cube_head_on"), "CENTER")
        self.assertIsNone(side_from_scenario("jaywalker"))

    def test_heatmap_alert_uses_near_row_peak(self) -> None:
        quiet = np.zeros((5, 5), dtype=np.float32)
        loud = np.zeros((5, 5), dtype=np.float32)
        loud[4, 0] = 0.9
        fired, side, peak = heatmap_alert({"1s": loud, "2s": quiet}, cutoff=0.5)
        self.assertTrue(fired)
        self.assertEqual(side, "LEFT")
        self.assertGreater(peak, 0.8)
        fired2, _, _ = heatmap_alert({"1s": quiet}, cutoff=0.5)
        self.assertFalse(fired2)


if __name__ == "__main__":
    unittest.main()
