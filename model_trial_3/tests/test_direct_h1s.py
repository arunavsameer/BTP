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
from collision_jepa3.training.checkpoint import extract_teacher_state, load_teacher_state
from collision_jepa3.training.engine import composite_score


def _direct_cfg() -> Config:
    cfg = Config.load()
    cfg.set("heads.mode", "direct_h1s")
    return cfg


class DirectH1sTests(unittest.TestCase):
    def test_outputs_only_z_and_h1s(self) -> None:
        cfg = _direct_cfg()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        out = m(torch.rand(2, 16, 3, 64, 64))
        self.assertEqual(set(out), {"Z_t", "Hhat_1s"})
        self.assertEqual(tuple(out["Z_t"].shape), (2, 32, 5, 5))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (2, 5, 5))
        self.assertGreaterEqual(float(out["Hhat_1s"].detach().min()), 0.0)
        self.assertLessEqual(float(out["Hhat_1s"].detach().max()), 1.0)
        self.assertEqual(len(m.predictors), 0)
        self.assertEqual(len(m.delta_heads), 0)

    def test_loss_is_h1s_plus_aux(self) -> None:
        cfg = _direct_cfg()
        loss_fn = TeacherLoss(cfg)
        out = {
            "Hhat_1s": torch.rand(2, 5, 5, requires_grad=True),
            "Z_t": torch.randn(2, 32, 5, 5, requires_grad=True),
        }
        batch = {
            "h_1s": torch.rand(2, 5, 5),
            "amb_1s": torch.zeros(2, dtype=torch.bool),
        }
        packed = loss_fn(out, batch)
        packed.total.backward()
        self.assertTrue(torch.isfinite(packed.total))
        for key in ("L_total", "L_1s", "L_future", "L_FP", "L_dir", "L_bg"):
            self.assertIn(key, packed.terms)
        for key in ("L_now", "L_delta", "L_1_5s", "L_2s"):
            self.assertNotIn(key, packed.terms)
        self.assertTrue(out["Hhat_1s"].grad is not None)

    def test_selection_uses_1s_only(self) -> None:
        cfg = _direct_cfg()
        metrics = {
            "1s": {"high_threat_recall": 0.4, "stop_f1": 0.5, "nuisance": 0.1, "miss": 0.2},
            "1_5s": {"high_threat_recall": 1.0, "stop_f1": 1.0, "nuisance": 1.0, "miss": 1.0},
        }
        sel = composite_score(metrics, cfg)
        self.assertAlmostEqual(sel["future_recall"], 0.4, places=5)
        self.assertAlmostEqual(sel["stop_f1"], 0.5, places=5)
        self.assertAlmostEqual(sel["nuisance"], 0.1, places=5)
        self.assertAlmostEqual(sel["miss"], 0.2, places=5)

    def test_checkpoint_roundtrip(self) -> None:
        cfg = _direct_cfg()
        torch.manual_seed(0)
        b = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=b)
        x = torch.rand(1, 16, 3, 64, 64)
        with torch.no_grad():
            a = m(x)["Hhat_1s"].clone()
        blob = extract_teacher_state(m)
        m2 = CollisionTeacher3(cfg, backbone=DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16))
        load_teacher_state(m2, blob)
        m2.backbone.load_state_dict(m.backbone.state_dict())
        with torch.no_grad():
            c = m2(x)["Hhat_1s"]
        self.assertTrue(torch.allclose(a, c, atol=1e-5))

    def test_frozen_backbone_still_frozen(self) -> None:
        cfg = _direct_cfg()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        out = m(torch.rand(1, 16, 3, 64, 64))
        batch = {"h_1s": torch.rand(1, 5, 5), "amb_1s": torch.zeros(1, dtype=torch.bool)}
        TeacherLoss(cfg)(out, batch).total.backward()
        for p in m.backbone.parameters():
            self.assertFalse(p.requires_grad)
            self.assertIsNone(p.grad)
        adapter_grads = [p.grad for p in m.adapter_parameters()]
        self.assertTrue(any(g is not None for g in adapter_grads))


if __name__ == "__main__":
    unittest.main()
