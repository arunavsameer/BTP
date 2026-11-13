from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.evaluation.operating_point import operating_metrics, recall_at_false_alarm


class OperatingPointTests(unittest.TestCase):
    def test_perfect_scores_full_recall_at_low_fa(self) -> None:
        gt = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        pred = np.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.9])
        valid = np.ones_like(gt, dtype=bool)
        rec = recall_at_false_alarm(pred, gt, valid, fa_rate=0.05, hi=0.5)
        self.assertEqual(rec["recall_at_fa"], 1.0)

    def test_operating_keys(self) -> None:
        rng = np.random.RandomState(0)
        pred = rng.rand(8, 5, 5)
        gt = rng.rand(8, 5, 5)
        valid = np.ones((8, 5, 5), dtype=bool)
        out = operating_metrics(pred, gt, valid)
        for k in ("recall_at_5fa", "ap_high", "mean_pred_gt80", "mse", "fa_at_0.5"):
            self.assertIn(k, out)
            self.assertTrue(np.isfinite(out[k]) or np.isnan(out["mean_pred_gt80"]))


if __name__ == "__main__":
    unittest.main()
