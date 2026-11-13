"""Gate 1: param count, causality, frozen-decoder grads for CollisionStudentV2."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.parent import load_parent_decoder
from student.paths import parent_ckpt
from student.v2.data import hazard_sample_weights, hflip_batch
from student.v2.losses import get_lambda_feat, student_v2_loss
from student.v2.model import CollisionStudentV2, n_params


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


class StudentV2Tests(unittest.TestCase):
    def test_trainable_param_count_in_gate(self):
        model = CollisionStudentV2(DummyDecoder())
        n = n_params(model, trainable=True)
        self.assertGreaterEqual(n, 375_000)
        self.assertLessEqual(n, 385_000)
        self.assertFalse(any(p.requires_grad for p in model.decoder.parameters()))

    def test_causality_no_future_leak_into_t0_t6(self):
        model = CollisionStudentV2(DummyDecoder()).eval()
        clip_a = torch.rand(1, 16, 3, 256, 256)
        clip_b = clip_a.clone()
        clip_b[:, 14:] = torch.rand_like(clip_b[:, 14:])
        with torch.no_grad():
            xa = model.normalize(clip_a).permute(0, 2, 1, 3, 4)
            xb = model.normalize(clip_b).permute(0, 2, 1, 3, 4)
            ma = model.backbone(xa)["F_motion"]
            mb = model.backbone(xb)["F_motion"]
        err = (ma[:, :, :7] - mb[:, :, :7]).abs().max().item()
        self.assertLess(err, 1e-6, f"future leak {err}")

    def test_loss_flows_through_frozen_decoder(self):
        ckpt = parent_ckpt()
        decoder = load_parent_decoder(ckpt) if ckpt.is_file() else DummyDecoder()
        model = CollisionStudentV2(decoder)
        clip = torch.rand(2, 16, 3, 256, 256)
        ft = torch.randn(2, 128, 16, 16)
        pt = torch.zeros(2, 5, 5)
        pt[0, 4, 2] = 0.9
        out = model(clip)
        with torch.no_grad():
            z_t = model.decoder.forward_fine_logits(ft)
        batch = {"F_teacher": ft, "Z_teacher_16": z_t, "P_teacher_5": pt}
        loss, terms = student_v2_loss(out, batch)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.backbone.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
        self.assertIn("L_5x5", terms)
        self.assertIn("L_16x16", terms)
        self.assertIn("L_feat", terms)
        self.assertEqual(tuple(out["Z_student_16"].shape)[-2:], (16, 16))
        self.assertEqual(tuple(out["P_student_5"].shape)[-2:], (5, 5))

    def test_hazard_weights_and_hflip(self):
        w = hazard_sample_weights(torch.tensor([0.1, 0.4, 0.8]))
        self.assertEqual(w.tolist(), [1.0, 1.5, 3.0])
        batch = {
            "clip": torch.arange(16 * 3 * 4 * 4, dtype=torch.float32).view(1, 16, 3, 4, 4),
            "P_teacher_5": torch.arange(25, dtype=torch.float32).view(1, 5, 5),
        }
        flipped = hflip_batch({k: v.clone() for k, v in batch.items()})
        self.assertTrue(torch.equal(flipped["clip"], batch["clip"].flip(-1)))
        self.assertTrue(torch.equal(flipped["P_teacher_5"], batch["P_teacher_5"].flip(-1)))

    def test_lambda_feat_cosine_ramp(self):
        self.assertAlmostEqual(get_lambda_feat(20), 0.500, places=3)
        self.assertAlmostEqual(get_lambda_feat(26), 0.439, places=3)
        self.assertAlmostEqual(get_lambda_feat(32), 0.289, places=3)
        self.assertAlmostEqual(get_lambda_feat(39), 0.111, places=3)
        self.assertAlmostEqual(get_lambda_feat(45), 0.050, places=3)
        self.assertAlmostEqual(get_lambda_feat(46), 0.050, places=3)


if __name__ == "__main__":
    unittest.main()
