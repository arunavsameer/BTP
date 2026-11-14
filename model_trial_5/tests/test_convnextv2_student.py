#!/usr/bin/env python3
"""CPU unit tests for the 3D ConvNeXt-V2 sparse student."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.models.heatmap_decoder import HeatmapDecoder  # noqa: E402
from student_convnextv2.blocks import ConvNeXtV2Block3d  # noqa: E402
from student_convnextv2.losses import convnext_student_loss  # noqa: E402
from student_convnextv2.model import CollisionStudentConvNeXtV2, SparseConvNeXtV2_3D  # noqa: E402
from student_convnextv2.sparse import apply_submanifold, random_patch_mask  # noqa: E402


def dummy_decoder() -> HeatmapDecoder:
    dec = HeatmapDecoder(z_channels=128, dilated=True, out_grid=5, pool="max")
    for p in dec.parameters():
        p.requires_grad_(False)
    dec.eval()
    return dec


class SparseConvTests(unittest.TestCase):
    def test_mask_ratio_and_shape(self):
        m = random_patch_mask(4, 16, mask_ratio=0.6, device="cpu")
        self.assertEqual(tuple(m.shape), (4, 1, 1, 16, 16))
        frac = float((m == 0).float().mean())
        self.assertGreater(frac, 0.5)
        self.assertLess(frac, 0.7)

    def test_submanifold_zeros_masked(self):
        x = torch.ones(1, 2, 2, 4, 4)
        mask = torch.zeros(1, 1, 1, 4, 4)
        mask[..., :2, :2] = 1
        y = apply_submanifold(x, mask)
        self.assertEqual(float(y[..., 0, 0].sum()), 4.0)
        self.assertEqual(float(y[..., 3, 3].sum()), 0.0)

    def test_block_mask_ones_matches_dense(self):
        torch.manual_seed(0)
        blk = ConvNeXtV2Block3d(8, drop_path=0.0)
        x = torch.randn(2, 8, 2, 8, 8)
        mask = torch.ones(2, 1, 2, 8, 8)
        blk.eval()
        with torch.no_grad():
            a = blk(x, None)
            b = blk(x, mask)
        self.assertTrue(torch.allclose(a, b, atol=1e-5, rtol=1e-4))


class StudentShapeTests(unittest.TestCase):
    def test_atto_shapes_and_frozen_decoder(self):
        torch.manual_seed(0)
        dec = dummy_decoder()
        model = CollisionStudentConvNeXtV2(
            dec, variant="atto", mask_ratio=0.6, sparse_prob=1.0, grad_checkpoint=False
        )
        model.eval()
        clip = torch.rand(1, 16, 3, 256, 256)
        with torch.no_grad():
            out = model(clip, mask=None)
        self.assertEqual(tuple(out["F"].shape), (1, 128, 16, 16))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (1, 5, 5))
        self.assertEqual(tuple(out["Hhat_1s_logits"].shape), (1, 5, 5))
        audit = model.param_audit()
        self.assertEqual(audit["decoder"], 47809)
        self.assertGreater(audit["trainable"], 1_000_000)
        self.assertTrue(all(not p.requires_grad for p in model.decoder.parameters()))

    def test_loss_backward_skips_decoder(self):
        torch.manual_seed(0)
        dec = dummy_decoder()
        model = CollisionStudentConvNeXtV2(
            dec, variant="atto", sparse_prob=0.0, grad_checkpoint=False
        )
        model.train()
        clip = torch.rand(2, 16, 3, 256, 256)
        out = model(clip, mask=None)
        batch = {
            "F_teacher": torch.randn(2, 128, 16, 16),
            "H_teacher": torch.rand(2, 5, 5),
            "gt_1s": torch.rand(2, 5, 5),
            "h_now": torch.rand(2, 5, 5),
            "amb_1s": torch.zeros(2, dtype=torch.bool),
        }
        mu = torch.zeros(128, 1, 1)
        sigma = torch.ones(128, 1, 1)
        loss, terms = convnext_student_loss(out, batch, mu, sigma)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("L_feat", terms)
        loss.backward()
        g_dec = sum(float(p.grad.pow(2).sum()) for p in model.decoder.parameters() if p.grad is not None)
        g_stu = sum(float(p.grad.pow(2).sum()) for p in model.student_parameters() if p.grad is not None)
        self.assertEqual(g_dec, 0.0)
        self.assertGreater(g_stu, 0.0)

    def test_sparse_forward_keeps_shape(self):
        torch.manual_seed(1)
        backbone = SparseConvNeXtV2_3D(
            depths=[2, 2, 6, 2],
            dims=[40, 80, 160, 320],
            drop_path=0.0,
            mask_ratio=0.6,
            sparse_prob=1.0,
            grad_checkpoint=False,
        )
        backbone.eval()
        x = torch.randn(1, 3, 16, 256, 256)
        mask = random_patch_mask(1, 16, mask_ratio=0.6, device="cpu")
        with torch.no_grad():
            out = backbone(x, mask=mask)
        self.assertEqual(tuple(out["F"].shape), (1, 128, 16, 16))
        self.assertIsNotNone(out["sparse_mask"])


if __name__ == "__main__":
    unittest.main()
