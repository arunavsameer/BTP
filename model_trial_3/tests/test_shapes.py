from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.models.spatial_adapter import SpatialTemporalAdapter
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone


class ShapeTests(unittest.TestCase):
    def test_adapter_z_shape(self) -> None:
        ad = SpatialTemporalAdapter(hidden_size=32, temporal_groups=8, spatial=4, z_channels=32, feature_grid=5, projection_channels=32)
        tokens = torch.randn(2, 8 * 4 * 4, 32)
        z = ad(tokens)
        self.assertEqual(tuple(z.shape), (2, 32, 5, 5))

    def test_teacher_outputs(self) -> None:
        cfg = Config.load()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        clip = torch.rand(2, 16, 3, 64, 64)
        out = m(clip)
        self.assertEqual(tuple(out["Z_t"].shape), (2, 32, 5, 5))
        for k in ("Hhat_now", "Hhat_1s", "Hhat_1_5s", "Hhat_2s"):
            self.assertEqual(tuple(out[k].shape), (2, 5, 5))
            self.assertGreaterEqual(float(out[k].detach().min()), 0.0)
            self.assertLessEqual(float(out[k].detach().max()), 1.0)
        for k in ("Delta_1s", "Delta_1_5s", "Delta_2s"):
            self.assertEqual(tuple(out[k].shape), (2, 5, 5))
            self.assertGreaterEqual(float(out[k].detach().min()), -1.0)
            self.assertLessEqual(float(out[k].detach().max()), 1.0)

    def test_direct_h1s_outputs(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        out = m(torch.rand(2, 16, 3, 64, 64))
        self.assertEqual(set(out), {"Z_t", "Hhat_1s"})
        self.assertEqual(tuple(out["Hhat_1s"].shape), (2, 5, 5))

    def test_wrong_token_count_fails(self) -> None:
        ad = SpatialTemporalAdapter(hidden_size=32, temporal_groups=8, spatial=4)
        tokens = torch.randn(1, 100, 32)
        with self.assertRaises(RuntimeError):
            ad(tokens)


if __name__ == "__main__":
    unittest.main()
