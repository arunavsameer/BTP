from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.training.early_stopping import EarlyStopping
from collision_jepa3.training.engine import (
    horizon_mse,
    balanced_warning_better,
    prefork_stop_better,
    recall5_better,
    recall5_operating_valid,
)


class ValMseStopTests(unittest.TestCase):
    def test_horizon_mse_uses_supervised_1s(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        val = {
            "horizons": {
                "1s": {"mse": 0.04},
                "1_5s": {"mse": 9.0},
            }
        }
        self.assertAlmostEqual(horizon_mse(val, cfg), 0.04)

    def test_three_consecutive_non_falls_restore_x_minus_3(self) -> None:
        stopper = EarlyStopping(patience=3)
        prev = None
        best_epoch = None
        stop_epoch = None
        series = [0.10, 0.08, 0.07, 0.075, 0.08, 0.09]
        for epoch, mse in enumerate(series):
            better = prev is None or mse < prev
            prev = mse
            if better:
                best_epoch = epoch
            if stopper.step(better):
                stop_epoch = epoch
                break
        self.assertEqual(stop_epoch, 5)
        self.assertEqual(best_epoch, 2)
        self.assertEqual(stop_epoch - 3, best_epoch)

    def test_recall5_rejects_all_zero_maps(self) -> None:
        collapsed = {
            "recall_at_5fa": 1.0,
            "fa_achieved_5fa": 1.0,
            "mean_pred_gt80": 0.0,
        }
        healthy = {
            "recall_at_5fa": 0.82,
            "fa_achieved_5fa": 0.05,
            "mean_pred_gt80": 0.68,
        }
        self.assertFalse(recall5_operating_valid(collapsed))
        self.assertTrue(recall5_operating_valid(healthy))
        self.assertTrue(recall5_better(healthy, None))
        self.assertFalse(recall5_better(collapsed, healthy))
        worse = dict(healthy)
        worse["recall_at_5fa"] = 0.70
        self.assertFalse(recall5_better(worse, healthy))
        better = dict(healthy)
        better["recall_at_5fa"] = 0.90
        self.assertTrue(recall5_better(better, healthy))

    def test_prefork_stop_requires_post_floor_then_pre_gain(self) -> None:
        low_post = {"pre_fork_stop_recall": 0.70, "post_fork_stop_recall": 0.80}
        climb = {"pre_fork_stop_recall": 0.60, "post_fork_stop_recall": 0.91}
        better_pre = {"pre_fork_stop_recall": 0.65, "post_fork_stop_recall": 0.92}
        worse_pre = {"pre_fork_stop_recall": 0.58, "post_fork_stop_recall": 0.95}
        drop_post = {"pre_fork_stop_recall": 0.80, "post_fork_stop_recall": 0.85}
        self.assertTrue(prefork_stop_better(low_post, None, post_floor=0.90))
        self.assertTrue(prefork_stop_better(climb, low_post, post_floor=0.90))
        self.assertTrue(prefork_stop_better(better_pre, climb, post_floor=0.90))
        self.assertFalse(prefork_stop_better(worse_pre, better_pre, post_floor=0.90))
        self.assertFalse(prefork_stop_better(drop_post, better_pre, post_floor=0.90))

    def test_balanced_warning_rejects_safe_stop_spike(self) -> None:
        good = {
            "safe_recall": 0.86,
            "stop_recall": 0.80,
            "post_fork_stop_recall": 0.94,
            "safe_stop_rate": 0.05,
        }
        louder = {
            "safe_recall": 0.74,
            "stop_recall": 0.88,
            "post_fork_stop_recall": 0.97,
            "safe_stop_rate": 0.12,
        }
        balanced = {
            "safe_recall": 0.85,
            "stop_recall": 0.83,
            "post_fork_stop_recall": 0.93,
            "safe_stop_rate": 0.06,
        }
        self.assertTrue(balanced_warning_better(good, None))
        self.assertFalse(balanced_warning_better(louder, good, max_safe_stop=0.08, safe_floor=0.82))
        self.assertTrue(balanced_warning_better(balanced, good, max_safe_stop=0.08))

    def test_balanced_warning_allows_post_just_under_090_when_floor_088(self) -> None:
        ep0 = {
            "safe_recall": 0.946,
            "stop_recall": 0.678,
            "post_fork_stop_recall": 0.902,
            "safe_stop_rate": 0.015,
        }
        ep5 = {
            "safe_recall": 0.932,
            "stop_recall": 0.707,
            "post_fork_stop_recall": 0.893,
            "safe_stop_rate": 0.016,
        }
        self.assertFalse(balanced_warning_better(ep5, ep0, post_floor=0.90))
        self.assertTrue(balanced_warning_better(ep5, ep0, post_floor=0.88, stop_floor=0.67))


if __name__ == "__main__":
    unittest.main()
