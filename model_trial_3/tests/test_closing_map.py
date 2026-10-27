from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.closing_map import load_closing_maps, maps_from_frames, object_approaching
from collision_jepa3.losses.composite import TeacherLoss


def _obj(cls, ttc, rvz, bbox, threat="SAFE_DYNAMIC"):
    return {
        "class_name": cls,
        "threat_label": threat,
        "kinematics": {"ttc": ttc, "relative_velocity": [0.0, 0.0, rvz], "distance": 5.0},
        "bounding_box_2d": bbox,
    }


class ClosingMapTests(unittest.TestCase):
    def test_hold_person_not_approaching(self) -> None:
        self.assertFalse(object_approaching(_obj("person", 9999.0, 0.003, {})))

    def test_pulling_person_not_approaching(self) -> None:
        self.assertFalse(object_approaching(_obj("person", 9999.0, 0.34, {})))

    def test_closing_person_approaching(self) -> None:
        self.assertTrue(object_approaching(_obj("person", 5.5, -0.38, {})))

    def test_jaywalker_approaching(self) -> None:
        self.assertTrue(object_approaching(_obj("person", 1.0, -2.1, {}, threat="CRITICAL_THREAT")))

    def test_trees_ignored_person_splatted(self) -> None:
        frames = [
            {
                "frame_id": 0,
                "objects": [
                    _obj("tree", 20.0, -1.0, {"xmin": 0, "ymin": 0, "xmax": 1920, "ymax": 1080}),
                    _obj("person", 9999.0, 0.0, {"xmin": 960, "ymin": 540, "xmax": 1200, "ymax": 1080}),
                    _obj("person", 2.0, -1.5, {"xmin": 0, "ymin": 0, "xmax": 400, "ymax": 400}),
                ],
            }
        ]
        close, occ = maps_from_frames(frames, expected_frames=1)
        self.assertGreater(float(occ[0].max()), 0.0)
        self.assertGreater(float(close[0, 0, 0]), 0.0)
        self.assertEqual(float(close[0, 4, 4]), 0.0)
        self.assertGreater(float(occ[0, 4, 3]), 0.0)

    def test_missing_file_zeros(self) -> None:
        close, occ = load_closing_maps("/no/such/annotations.json")
        self.assertEqual(close.shape, (450, 5, 5))
        self.assertEqual(float(close.max()), 0.0)
        self.assertEqual(float(occ.max()), 0.0)

    def test_load_roundtrip(self) -> None:
        frames = [{"frame_id": "000010", "objects": [_obj("vehicle", 0.7, -1.2, {"xmin": 0, "ymin": 800, "xmax": 1920, "ymax": 1080})]}]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "annotations.json"
            path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            close, occ = load_closing_maps(path, expected_frames=20)
            self.assertGreater(float(close[10, 4].max()), 0.0)
            self.assertEqual(float(close[0].max()), 0.0)

    def test_composite_close_loss_on_occupied(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("loss.close_weight", 1.0)
        pred = torch.full((1, 5, 5), 0.01)
        close_hat = torch.full((1, 5, 5), 0.01)
        tgt = torch.zeros(1, 5, 5)
        h_close = torch.zeros(1, 5, 5)
        h_close[0, 2, 2] = 1.0
        mask = torch.zeros(1, 5, 5)
        mask[0, 2, 2] = 1.0
        batch = {
            "h_1s": tgt,
            "h_now": tgt,
            "h_close": h_close,
            "close_mask": mask,
            "amb_1s": torch.zeros(1, dtype=torch.bool),
        }
        out = {
            "Hhat_1s": pred,
            "Hhat_close": close_hat,
            "Z_t": torch.zeros(1, 32, 5, 5),
        }
        missed = TeacherLoss(cfg)(out, batch)
        close_hat2 = close_hat.clone()
        close_hat2[0, 2, 2] = 0.95
        out2 = {"Hhat_1s": pred, "Hhat_close": close_hat2, "Z_t": torch.zeros(1, 32, 5, 5)}
        hit = TeacherLoss(cfg)(out2, batch)
        self.assertGreater(float(missed.terms["L_close"]), float(hit.terms["L_close"]))
        self.assertGreater(float(missed.total), float(hit.total))


if __name__ == "__main__":
    unittest.main()
