from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.sampling import (
    FRAME_OFFSETS,
    available_horizons,
    input_indices,
    iter_anchors,
    target_indices,
    valid_anchor_range,
)


class SamplingTests(unittest.TestCase):
    def test_locked_offsets(self) -> None:
        self.assertEqual(len(FRAME_OFFSETS), 16)
        self.assertEqual(FRAME_OFFSETS[-1], 0)
        self.assertEqual(FRAME_OFFSETS[0], -90)
        diffs = [FRAME_OFFSETS[i + 1] - FRAME_OFFSETS[i] for i in range(15)]
        self.assertTrue(all(d == 6 for d in diffs))

    def test_input_indices_exact(self) -> None:
        idx = input_indices(90)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[-1], 90)
        self.assertEqual(idx, [90 + o for o in FRAME_OFFSETS])

    def test_horizons(self) -> None:
        tg = target_indices(100, [30, 45, 60])
        self.assertEqual(tg["now"], 100)
        self.assertEqual(tg["1s"], 130)
        self.assertEqual(tg["1_5s"], 145)
        self.assertEqual(tg["2s"], 160)

    def test_valid_range_450_strict(self) -> None:
        t_min, t_max = valid_anchor_range(450, allow_partial_horizons=False)
        self.assertEqual(t_min, 90)
        self.assertEqual(t_max, 389)
        anchors = iter_anchors(450, 15, allow_partial_horizons=False)
        self.assertEqual(anchors[0], 90)
        self.assertEqual(anchors[-1], 375)
        self.assertNotIn(420, anchors)

    def test_valid_range_450_partial_tmax_419(self) -> None:
        t_min, t_max = valid_anchor_range(450, allow_partial_horizons=True)
        self.assertEqual(t_min, 90)
        self.assertEqual(t_max, 419)
        anchors = iter_anchors(450, 15, allow_partial_horizons=True)
        self.assertEqual(anchors[0], 90)
        self.assertEqual(anchors[-1], 405)
        self.assertIn(390, anchors)
        self.assertIn(405, anchors)
        self.assertNotIn(420, anchors)

    def test_partial_horizons_mask_missing_futures(self) -> None:
        avail_405 = available_horizons(405, 450)
        self.assertTrue(avail_405["now"])
        self.assertTrue(avail_405["1s"])
        self.assertFalse(avail_405["1_5s"])
        self.assertFalse(avail_405["2s"])
        avail_419 = available_horizons(419, 450)
        self.assertTrue(avail_419["now"])
        self.assertTrue(avail_419["1s"])
        self.assertFalse(avail_419["1_5s"])
        self.assertFalse(avail_419["2s"])
        avail_420 = available_horizons(420, 450)
        self.assertTrue(avail_420["now"])
        self.assertFalse(avail_420["1s"])
        self.assertFalse(avail_420["1_5s"])
        self.assertFalse(avail_420["2s"])
        avail_300 = available_horizons(300, 450)
        self.assertTrue(all(avail_300.values()))


if __name__ == "__main__":
    unittest.main()
