from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone


class FrozenBackboneTests(unittest.TestCase):
    def test_frozen_has_no_backbone_grads(self) -> None:
        cfg = Config.load()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        clip = torch.rand(1, 16, 3, 64, 64)
        out = m(clip)
        batch = {
            "h_now": torch.rand(1, 5, 5),
            "h_1s": torch.rand(1, 5, 5),
            "h_1_5s": torch.rand(1, 5, 5),
            "h_2s": torch.rand(1, 5, 5),
            "amb_1s": torch.zeros(1, dtype=torch.bool),
            "amb_1_5s": torch.zeros(1, dtype=torch.bool),
            "amb_2s": torch.zeros(1, dtype=torch.bool),
        }
        TeacherLoss(cfg)(out, batch).total.backward()
        for p in m.backbone.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        adapter_grads = [p.grad for p in m.adapter_parameters()]
        self.assertTrue(any(g is not None for g in adapter_grads))
        self.assertTrue(all(torch.isfinite(g).all() for g in adapter_grads if g is not None))


if __name__ == "__main__":
    unittest.main()
