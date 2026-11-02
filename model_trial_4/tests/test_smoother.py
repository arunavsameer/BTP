"""Consistency mixer: raw-window std, clip resets, and a learnable tau."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa4.config import Config
from collision_jepa4.losses.composite import TeacherLoss
from collision_jepa4.models.smoother import ConsistencySmoother, occupancy_weighted_std
from collision_jepa4.models.teacher import CollisionTeacher4
from collision_jepa4.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa4.training.checkpoint import extract_teacher_state, load_teacher_state


def _teacher(enabled: bool = True) -> CollisionTeacher4:
    cfg = Config.load()
    cfg.set("smoother.enabled", enabled)
    backbone = DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=256,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        num_layers=1,
    )
    return CollisionTeacher4(cfg, backbone=backbone)


class SmootherTests(unittest.TestCase):
    def test_constant_window_has_zero_std(self) -> None:
        maps = torch.full((4, 5, 5), 0.4)
        self.assertEqual(float(occupancy_weighted_std(maps)), 0.0)
        self.assertEqual(float(occupancy_weighted_std(maps[:1])), 0.0)

    def test_first_snapshot_copies_raw(self) -> None:
        mix = ConsistencySmoother(window=4, tau_init=0.5)
        raw = torch.rand(1, 5, 5).clamp(0.05, 0.95)
        out = mix(raw, clip_ids=["clip_a"])
        self.assertTrue(torch.allclose(out["Hhat_1s"], raw))
        self.assertEqual(float(out["alpha"].detach()), 1.0)

    def test_identical_raw_maps_keep_alpha_at_one(self) -> None:
        mix = ConsistencySmoother(window=4, tau_init=0.5)
        raw = torch.full((4, 5, 5), 0.6)
        out = mix(raw, clip_ids=["c"] * 4)
        self.assertTrue(torch.allclose(out["Hhat_1s"], raw, atol=1e-6))
        self.assertTrue(torch.allclose(out["alpha"], torch.ones(4), atol=1e-5))

    def test_flicker_lowers_alpha(self) -> None:
        mix = ConsistencySmoother(window=4, tau_init=0.5)
        a = torch.zeros(5, 5)
        b = torch.ones(5, 5)
        raw = torch.stack([a, b, a, b], dim=0)
        out = mix(raw, clip_ids=["c"] * 4)
        self.assertEqual(float(out["alpha"][0].detach()), 1.0)
        self.assertLess(float(out["alpha"][3].detach()), 0.5)

    def test_clip_id_change_resets(self) -> None:
        mix = ConsistencySmoother(window=4, tau_init=0.5)
        raw = torch.stack([torch.zeros(5, 5), torch.ones(5, 5)], dim=0)
        out = mix(raw, clip_ids=["a", "b"])
        self.assertTrue(torch.allclose(out["Hhat_1s"][0], raw[0]))
        self.assertTrue(torch.allclose(out["Hhat_1s"][1], raw[1]))
        self.assertEqual(float(out["alpha"][0].detach()), 1.0)
        self.assertEqual(float(out["alpha"][1].detach()), 1.0)

    def test_tau_receives_grad(self) -> None:
        mix = ConsistencySmoother(window=4, tau_init=0.5)
        a = torch.zeros(5, 5)
        b = torch.ones(5, 5)
        raw = torch.stack([a, b, a, b], dim=0)
        out = mix(raw, clip_ids=["c"] * 4)
        out["Hhat_1s"].sum().backward()
        self.assertIsNotNone(mix.log_tau.grad)
        self.assertTrue(torch.isfinite(mix.log_tau.grad))
        self.assertGreater(float(mix.log_tau.grad.abs()), 0.0)

    def test_teacher_loss_uses_mixed_map(self) -> None:
        model = _teacher()
        tokens = torch.randn(4, 2048, 1408)
        out = model(tokens=tokens, clip_ids=["c"] * 4)
        self.assertEqual(tuple(out["Hhat_1s"].shape), (4, 5, 5))
        self.assertEqual(tuple(out["alpha"].shape), (4,))
        self.assertTrue(torch.allclose(out["Hhat_1s"][0], out["Hhat_1s_raw"][0]))
        self.assertEqual(float(out["alpha"][0].detach()), 1.0)
        loss = TeacherLoss(Config.load())(out, {"h_1s": torch.rand(4, 5, 5)})
        loss.total.backward()
        self.assertIsNotNone(model.smoother.log_tau.grad)
        self.assertIsNotNone(model.trunk.projection.weight.grad)

    def test_disabled_mixer_is_identity(self) -> None:
        model = _teacher(enabled=False)
        tokens = torch.randn(3, 2048, 1408)
        out = model(tokens=tokens, clip_ids=["c"] * 3)
        self.assertTrue(torch.equal(out["Hhat_1s"], out["Hhat_1s_raw"]))
        self.assertTrue(torch.equal(out["Hhat_1s_logits"], out["Hhat_1s_raw_logits"]))

    def test_checkpoint_roundtrip_keeps_tau(self) -> None:
        model = _teacher()
        with torch.no_grad():
            model.smoother.log_tau.fill_(1.25)
        blob = extract_teacher_state(model)
        other = _teacher()
        load_teacher_state(other, blob)
        self.assertAlmostEqual(float(other.smoother.log_tau.detach()), 1.25, places=5)


if __name__ == "__main__":
    unittest.main()
