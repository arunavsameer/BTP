from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.evaluation.labels import DEFAULT_WARNING, heatmap_near_row_max
from collision_jepa3.evaluation.metrics import SAFE, CAUTION, STOP, classify_heatmap


class WarningTierTests(unittest.TestCase):
    def test_config_matches_new_tiers(self) -> None:
        cfg = Config.load()
        self.assertEqual(cfg.get("warning.caution_threshold"), 0.25)
        self.assertEqual(cfg.get("warning.stop_threshold"), 0.60)
        self.assertEqual(list(cfg.get("warning.row_weights")), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(list(cfg.get("adapter.row_weights")), [1.0, 1.0, 1.0, 1.0, 1.0])
        self.assertEqual(cfg.get("loss.direction_threat_threshold"), 0.25)

    def test_plain_max_far_row_can_stop(self) -> None:
        hm = np.zeros((5, 5), dtype=np.float32)
        hm[0, 2] = 0.65
        direction, sev, _ = classify_heatmap(hm, DEFAULT_WARNING)
        self.assertEqual(direction, "CENTER")
        self.assertEqual(sev, STOP)
        self.assertAlmostEqual(heatmap_near_row_max(hm), 0.65, places=5)

    def test_bands(self) -> None:
        def sev(v):
            hm = np.zeros((5, 5), dtype=np.float32)
            hm[4, 2] = v
            return classify_heatmap(hm, DEFAULT_WARNING)[1]

        self.assertEqual(sev(0.24), SAFE)
        self.assertEqual(sev(0.25), CAUTION)
        self.assertEqual(sev(0.59), CAUTION)
        self.assertEqual(sev(0.60), STOP)


if __name__ == "__main__":
    unittest.main()
