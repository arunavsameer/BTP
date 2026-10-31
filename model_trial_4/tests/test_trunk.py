"""Shape, parameter, LayerNorm, and one-step BCE checks for the 1s-ahead trunk."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa4.config import Config
from collision_jepa4.losses.composite import TeacherLoss
from collision_jepa4.models.teacher import CollisionTeacher4
from collision_jepa4.models.trunk import TRUNK_PARAMS_1408, CollisionTrunk, channel_layernorm
from collision_jepa4.models.vjepa_backbone import DummyVJEPA2Backbone


def _teacher() -> CollisionTeacher4:
    cfg = Config.load()
    backbone = DummyVJEPA2Backbone(
        hidden_size=1408,
        crop_size=256,
        patch_size=16,
        tubelet_size=2,
        num_frames=16,
        num_layers=1,
    )
    return CollisionTeacher4(cfg, backbone=backbone)


class TrunkTests(unittest.TestCase):
    def test_stage_shapes_and_probabilities(self) -> None:
        trunk = CollisionTrunk()
        tokens = torch.randn(2, 2048, 1408)
        projected = trunk.project(tokens)
        self.assertEqual(tuple(projected.shape), (2, 128, 8, 16, 16))
        mixed = trunk.blocks(projected)
        self.assertEqual(tuple(mixed.shape), (2, 128, 8, 16, 16))
        down = trunk.temporal_downsample(mixed)
        self.assertEqual(tuple(down.shape), (2, 96, 4, 16, 16))
        fused = trunk.aggregate(down)
        self.assertEqual(tuple(fused.shape), (2, 48, 16, 16))
        out = trunk(tokens)
        self.assertEqual(tuple(out["logits_fine"].shape), (2, 1, 16, 16))
        self.assertEqual(tuple(out["Hhat_1s_logits"].shape), (2, 5, 5))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (2, 5, 5))
        probs = out["Hhat_1s"].detach()
        self.assertGreater(float(probs.min()), 0.0)
        self.assertLess(float(probs.max()), 1.0)

    def test_parameter_count(self) -> None:
        trunk = CollisionTrunk()
        n = sum(p.numel() for p in trunk.parameters())
        self.assertEqual(n, TRUNK_PARAMS_1408)

    def test_layernorm_is_per_location_over_channels(self) -> None:
        norm = nn.LayerNorm(128)
        x = torch.zeros(1, 128, 2, 2, 2)
        x[0, :, 0, 0, 0] = torch.arange(128, dtype=torch.float32)
        x[0, :, 1, 1, 1] = 5.0
        y = channel_layernorm(norm, x)
        loc = y[0, :, 0, 0, 0].detach()
        self.assertLess(float(loc.mean().abs()), 1e-5)
        self.assertLess(float((loc.std(unbiased=False) - 1).abs()), 1e-4)
        self.assertLess(float(y[0, :, 1, 1, 1].detach().abs().max()), 1e-5)
        alone = x.clone()
        alone[0, :, 1, 1, 1] = 0
        y_alone = channel_layernorm(norm, alone)
        self.assertTrue(torch.allclose(y[0, :, 0, 0, 0], y_alone[0, :, 0, 0, 0]))

    def test_wrong_token_count_raises(self) -> None:
        trunk = CollisionTrunk()
        with self.assertRaises(RuntimeError):
            trunk.project(torch.randn(1, 100, 1408))

    def test_frozen_backbone_and_projection_grad(self) -> None:
        model = _teacher()
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
        tokens = torch.randn(2, 2048, 1408)
        out = model(tokens=tokens)
        loss = TeacherLoss(Config.load())(out, {"h_1s": torch.rand(2, 5, 5)})
        loss.total.backward()
        self.assertIsNotNone(model.trunk.projection.weight.grad)
        self.assertTrue(torch.isfinite(model.trunk.projection.weight.grad).all())
        for p in model.backbone.parameters():
            self.assertIsNone(p.grad)
        clip = torch.rand(1, 16, 3, 256, 256)
        clip_out = model(clip=clip)
        self.assertEqual(tuple(clip_out["Hhat_1s"].shape), (1, 5, 5))

    def test_loss_reads_h1s_not_h_now(self) -> None:
        model = _teacher()
        tokens = torch.randn(1, 2048, 1408)
        out = model(tokens=tokens)
        loss_fn = TeacherLoss(Config.load())
        h1s = torch.ones(1, 5, 5)
        a = loss_fn(out, {"h_1s": h1s, "h_now": torch.zeros(1, 5, 5)})
        b = loss_fn(out, {"h_1s": h1s, "h_now": torch.ones(1, 5, 5)})
        self.assertAlmostEqual(float(a.total), float(b.total), places=5)
        self.assertIn("L_1s", a.as_floats())


if __name__ == "__main__":
    unittest.main()
