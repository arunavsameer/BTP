from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.heatmap_annotations import HeatmapError, load_heatmaps, targets_for_anchor


class HeatmapAlignmentTests(unittest.TestCase):
    def test_join_by_frame_id_not_list_order(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            frames = []
            for i in range(450):
                mat = np.full((5, 5), i / 449.0, dtype=np.float32)
                frames.append({"frame_id": f"{i:06d}", "matrix": mat.tolist()})
            frames = list(reversed(frames))
            path.write_text(json.dumps({"k": 5, "frames": frames}), encoding="utf-8")
            hm = load_heatmaps(path)
            self.assertEqual(hm.shape, (450, 5, 5))
            self.assertAlmostEqual(float(hm[0, 0, 0]), 0.0, places=5)
            self.assertAlmostEqual(float(hm[449, 0, 0]), 1.0, places=5)

    def test_orientation_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            frames = []
            for i in range(450):
                mat = np.zeros((5, 5), dtype=np.float32)
                mat[0, 0] = 0.25  # far-left
                mat[4, 4] = 0.75  # near-right
                frames.append({"frame_id": i, "matrix": mat.tolist()})
            path.write_text(json.dumps({"k": 5, "frames": frames}), encoding="utf-8")
            hm = load_heatmaps(path)
            self.assertAlmostEqual(float(hm[10, 0, 0]), 0.25)
            self.assertAlmostEqual(float(hm[10, 4, 4]), 0.75)

    def test_corrupt_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hm.json"
            path.write_text(json.dumps({"k": 5, "frames": [{"frame_id": 0, "matrix": [[1, 2]]}]}), encoding="utf-8")
            with self.assertRaises(HeatmapError):
                load_heatmaps(path, expected_frames=1)

    def test_targets_for_anchor(self) -> None:
        hm = np.zeros((450, 5, 5), dtype=np.float32)
        hm[100] = 0.1
        hm[130] = 0.2
        hm[145] = 0.3
        hm[160] = 0.4
        tg = targets_for_anchor(hm, 100, [30, 45, 60])
        self.assertAlmostEqual(float(tg["h_now"].mean()), 0.1)
        self.assertAlmostEqual(float(tg["h_1s"].mean()), 0.2)
        self.assertAlmostEqual(float(tg["h_1_5s"].mean()), 0.3)
        self.assertAlmostEqual(float(tg["h_2s"].mean()), 0.4)

    def test_targets_allow_missing_late_horizons(self) -> None:
        hm = np.zeros((450, 5, 5), dtype=np.float32)
        hm[420] = 0.5
        hm[435] = 0.7
        with self.assertRaises(HeatmapError):
            targets_for_anchor(hm, 420, [30, 45, 60])
        tg = targets_for_anchor(hm, 420, [30, 45, 60], allow_missing=True)
        self.assertAlmostEqual(float(tg["h_now"].mean()), 0.5)
        self.assertAlmostEqual(float(tg["h_1s"].mean()), 0.0)
        tg405 = targets_for_anchor(hm, 405, [30, 45, 60], allow_missing=True)
        self.assertAlmostEqual(float(tg405["h_1s"].mean()), 0.7)
        self.assertAlmostEqual(float(tg405["h_2s"].mean()), 0.0)


if __name__ == "__main__":
    unittest.main()
