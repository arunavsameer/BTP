"""Unit tests: Chebyshev dilation, false hot on BLUE, intent same vs opposite."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from metrics import (  # noqa: E402
    allowed_mask,
    classify_peak,
    dilate_chebyshev,
    intent_match,
    slope_sign,
    spatial_safe,
    spatial_stop,
)


class TestDilation(unittest.TestCase):
    def test_chebyshev_1_is_3x3(self) -> None:
        m = np.zeros((5, 5), dtype=bool)
        m[2, 2] = True
        d = dilate_chebyshev(m, 1)
        expect = np.zeros((5, 5), dtype=bool)
        expect[1:4, 1:4] = True
        np.testing.assert_array_equal(d, expect)

    def test_edge_cell(self) -> None:
        m = np.zeros((5, 5), dtype=bool)
        m[0, 0] = True
        d = dilate_chebyshev(m, 1)
        self.assertTrue(d[0, 0] and d[0, 1] and d[1, 0] and d[1, 1])
        self.assertFalse(d[0, 2] or d[2, 0] or d[2, 2])

    def test_allowed_includes_diagonals(self) -> None:
        gt = np.zeros((5, 5), dtype=np.float32)
        gt[2, 2] = 0.80
        allow = allowed_mask(gt, hot_threshold=0.50, radius=1)
        self.assertTrue(allow[1, 1] and allow[1, 3] and allow[3, 1] and allow[3, 3])
        self.assertFalse(allow[0, 0])


class TestFalseHotOnBlue(unittest.TestCase):
    def test_spread_in_neighborhood_is_not_an_error(self) -> None:
        gt = np.zeros((5, 5), dtype=np.float32)
        gt[2, 2] = 0.90
        pred = np.zeros((5, 5), dtype=np.float32)
        pred[2, 2] = 0.70
        pred[1, 1] = 0.65  # diagonal of the red cell, Chebyshev 1
        s = spatial_stop(pred, gt)
        self.assertGreaterEqual(s["coverage"], 1.0 - 1e-6)
        self.assertEqual(s["leak"], 0.0)
        self.assertGreaterEqual(s["spatial"], 1.0 - 1e-6)

    def test_false_hot_on_blue_is_penalized(self) -> None:
        gt = np.zeros((5, 5), dtype=np.float32)
        gt[2, 2] = 0.90
        pred = np.zeros((5, 5), dtype=np.float32)
        pred[2, 2] = 0.80
        pred[0, 4] = 0.60  # BLUE, far from the blob
        s = spatial_stop(pred, gt)
        self.assertGreaterEqual(s["coverage"], 1.0 - 1e-6)
        self.assertGreater(s["leak"], 0.0)
        self.assertLess(s["spatial"], s["coverage"])

    def test_missing_threat_lowers_coverage(self) -> None:
        gt = np.zeros((5, 5), dtype=np.float32)
        gt[2, 2] = 0.90
        pred = np.zeros((5, 5), dtype=np.float32)
        pred[2, 2] = 0.30  # in the neighborhood but below STOP peak
        s = spatial_stop(pred, gt)
        self.assertAlmostEqual(s["coverage"], 0.30 / 0.60, places=5)
        self.assertLess(s["spatial"], 1.0)

    def test_empty_allowed_on_stop_is_zero(self) -> None:
        gt = np.full((5, 5), 0.40, dtype=np.float32)  # THREAT cells, not core-red
        pred = np.full((5, 5), 0.80, dtype=np.float32)
        s = spatial_stop(pred, gt)
        self.assertEqual(s["n_allowed"], 0)
        self.assertEqual(s["spatial"], 0.0)

    def test_safe_cold_is_one(self) -> None:
        pred = np.full((5, 5), 0.10, dtype=np.float32)
        s = spatial_safe(pred)
        self.assertGreaterEqual(s["spatial"], 1.0 - 1e-6)

    def test_safe_false_hot_fraction(self) -> None:
        pred = np.zeros((5, 5), dtype=np.float32)
        pred[0, 0] = 0.40
        s = spatial_safe(pred)
        self.assertGreater(s["leak"], 0.0)
        self.assertLess(s["spatial"], 1.0)


class TestIntent(unittest.TestCase):
    def test_same_rising(self) -> None:
        g = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.70], dtype=np.float32)
        p = np.array([0.12, 0.18, 0.28, 0.41, 0.55, 0.68], dtype=np.float32)
        self.assertEqual(slope_sign(g), 1)
        self.assertEqual(intent_match(g, p), 1.0)

    def test_opposite_penalized(self) -> None:
        g = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.80], dtype=np.float32)
        p = np.array([0.80, 0.70, 0.50, 0.40, 0.20, 0.05], dtype=np.float32)
        self.assertEqual(slope_sign(g), 1)
        self.assertEqual(slope_sign(p), -1)
        self.assertEqual(intent_match(g, p), 0.0)

    def test_both_flat(self) -> None:
        g = np.array([0.20, 0.21, 0.19, 0.20, 0.22, 0.21], dtype=np.float32)
        p = np.array([0.18, 0.19, 0.17, 0.18, 0.20, 0.19], dtype=np.float32)
        self.assertEqual(slope_sign(g), 0)
        self.assertEqual(intent_match(g, p), 1.0)

    def test_one_flat_one_moving(self) -> None:
        g = np.array([0.20, 0.20, 0.20, 0.20, 0.20, 0.21], dtype=np.float32)
        p = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.70], dtype=np.float32)
        self.assertEqual(slope_sign(g), 0)
        self.assertEqual(slope_sign(p), 1)
        self.assertEqual(intent_match(g, p), 0.5)

    def test_deadzone_swallows_jitter(self) -> None:
        g = np.array([0.50, 0.51, 0.49, 0.50, 0.52, 0.52], dtype=np.float32)
        self.assertEqual(slope_sign(g, 0.05), 0)


class TestClassCuts(unittest.TestCase):
    def test_cuts(self) -> None:
        self.assertEqual(classify_peak(0.24), "SAFE")
        self.assertEqual(classify_peak(0.25), "THREAT")
        self.assertEqual(classify_peak(0.59), "THREAT")
        self.assertEqual(classify_peak(0.60), "STOP")


if __name__ == "__main__":
    unittest.main()
