"""V6: V2 backbone + ResidualGatedFusionV6. Isolated from V5 run."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.v2.model import CollisionStudentV2, n_params
from student.v6.losses import L5_SCALE, get_lambda_feat_v6, student_v6_loss
from student.v6.model import CollisionStudentV6, ResidualGatedFusionV6, decoder_lr


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(128, 1, kernel_size=1)

    def forward_fine_logits(self, z):
        return self.net(z)

    def forward_checkpoints(self, z):
        logits_fine = self.forward_fine_logits(z)
        pooled = torch.nn.functional.adaptive_max_pool2d(logits_fine, 5)
        p = torch.sigmoid(pooled).squeeze(1)
        return {"Z_16": logits_fine, "P_5": p, "Hhat_1s": p, "Hhat_1s_logits": pooled.squeeze(1)}


class StudentV6Tests(unittest.TestCase):
    def test_gate_bias_open(self):
        m = ResidualGatedFusionV6()
        self.assertAlmostEqual(float(torch.sigmoid(m.gate_conv.bias.detach()).mean()), 0.3775, places=3)

    def test_fusion_residual_and_shapes(self):
        m = ResidualGatedFusionV6()
        f = torch.randn(2, 96, 8, 16, 16)
        d1 = torch.randn(2, 96, 8, 16, 16)
        coords = torch.randn(2, 3, 8, 16, 16)
        out, g = m(f, d1, coords)
        self.assertEqual(tuple(out.shape), (2, 128, 8, 16, 16))
        self.assertEqual(tuple(g.shape), (2, 96, 8, 16, 16))

    def test_smaller_than_v4_larger_than_v2(self):
        v2 = CollisionStudentV2(DummyDecoder())
        v6 = CollisionStudentV6(DummyDecoder())
        n2 = n_params(v2, trainable=True)
        n6 = n_params(v6, trainable=True)
        self.assertGreater(n6, n2)
        self.assertLess(n6, 410_000)
        self.assertGreater(n6, 395_000)
        self.assertFalse(hasattr(v6.backbone, "vel_fuse"))
        self.assertFalse(any(p.requires_grad for p in v6.decoder.parameters()))

    def test_no_delta3_in_fusion(self):
        v6 = CollisionStudentV6(DummyDecoder())
        self.assertEqual(int(v6.backbone.gated_fusion.motion_proj[0].in_channels), 96)

    def test_causality(self):
        model = CollisionStudentV6(DummyDecoder()).eval()
        clip_a = torch.rand(1, 16, 3, 256, 256)
        clip_b = clip_a.clone()
        clip_b[:, 14:] = torch.rand_like(clip_b[:, 14:])
        with torch.no_grad():
            xa = model.normalize(clip_a).permute(0, 2, 1, 3, 4)
            xb = model.normalize(clip_b).permute(0, 2, 1, 3, 4)
            ma = model.backbone(xa)["F_motion"]
            mb = model.backbone(xb)["F_motion"]
        err = (ma[:, :, :7] - mb[:, :, :7]).abs().max().item()
        self.assertLess(err, 1e-5, f"future leak {err}")

    def test_lambda_and_decoder_lr(self):
        self.assertAlmostEqual(get_lambda_feat_v6(20), 0.50, places=3)
        self.assertAlmostEqual(get_lambda_feat_v6(45), 0.10, places=3)
        self.assertEqual(decoder_lr(20, 1.5e-3), 0.0)
        self.assertAlmostEqual(decoder_lr(21, 1.5e-3), 1.5e-3 * 0.15 * 0.5, places=8)

    def test_l5_scale_in_total(self):
        self.assertEqual(L5_SCALE, 8.0)
        b = 2
        out = {
            "Hhat_1s_logits": torch.zeros(b, 5, 5),
            "Z_student_16": torch.zeros(b, 1, 16, 16),
            "F_pred": torch.ones(b, 128, 16, 16),
        }
        batch = {
            "P_teacher_5": torch.zeros(b, 5, 5),
            "Z_teacher_16": torch.zeros(b, 1, 16, 16),
            "F_teacher": torch.ones(b, 128, 16, 16),
        }
        total, terms = student_v6_loss(out, batch, w_feat=0.5)
        recon = 8.0 * terms["L_5x5"] + terms["L_16x16"] + 0.5 * terms["L_feat"]
        self.assertAlmostEqual(float(total), recon, places=5)
        self.assertAlmostEqual(terms["L_5x5_contrib"], 8.0 * terms["L_5x5"], places=5)

    def test_gate_in_forward(self):
        model = CollisionStudentV6(DummyDecoder()).eval()
        with torch.no_grad():
            out = model(torch.rand(1, 16, 3, 256, 256))
        self.assertEqual(tuple(out["G_t"].shape), (1, 96, 8, 16, 16))
        self.assertIn("Z_student_16", out)


if __name__ == "__main__":
    unittest.main()
