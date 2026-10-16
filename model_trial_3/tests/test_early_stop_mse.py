from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.training.early_stopping import EarlyStopping
from collision_jepa3.training.engine import horizon_mse


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


if __name__ == "__main__":
    unittest.main()
