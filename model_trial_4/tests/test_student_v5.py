"""V5: same V4 backbone, new loss / lambda / decoder unfreeze. Isolated from V4 run dir."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.v4.model import CollisionStudentV4, n_params
from student.v5.losses import get_lambda_feat_v5, student_v5_loss
from student.v5.model import CollisionStudentV5, decoder_lr


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


class StudentV5Tests(unittest.TestCase):
    def test_lambda_floor_is_010(self):
        self.assertAlmostEqual(get_lambda_feat_v5(20), 0.50, places=3)
        self.assertAlmostEqual(get_lambda_feat_v5(21), 0.498, places=3)
        self.assertAlmostEqual(get_lambda_feat_v5(45), 0.10, places=3)
        self.assertAlmostEqual(get_lambda_feat_v5(46), 0.10, places=3)
        self.assertGreater(get_lambda_feat_v5(32), 0.10)

    def test_same_trainable_as_v4_while_decoder_frozen(self):
        v4 = CollisionStudentV4(DummyDecoder())
        v5 = CollisionStudentV5(DummyDecoder())
        self.assertEqual(n_params(v4, trainable=True), n_params(v5, trainable=True))
        self.assertFalse(v5.decoder_unfrozen)
        self.assertFalse(any(p.requires_grad for p in v5.decoder.parameters()))

    def test_decoder_unfreeze_and_train_mode(self):
        v5 = CollisionStudentV5(DummyDecoder())
        v5.set_decoder_trainable(True)
        self.assertTrue(v5.decoder_unfrozen)
        self.assertTrue(any(p.requires_grad for p in v5.decoder.parameters()))
        v5.train()
        self.assertTrue(v5.decoder.training)
        v5.set_decoder_trainable(False)
        v5.train()
        self.assertFalse(v5.decoder.training)

    def test_decoder_lr_ramp(self):
        self.assertEqual(decoder_lr(20, 1.5e-3), 0.0)
        self.assertAlmostEqual(decoder_lr(21, 1.5e-3), 1.5e-3 * 0.15 * 0.5, places=8)
        self.assertAlmostEqual(decoder_lr(22, 1.5e-3), 1.5e-3 * 0.15, places=8)
        self.assertAlmostEqual(decoder_lr(40, 1.0e-3), 1.0e-3 * 0.15, places=8)

    def test_v5_loss_backward_through_logits(self):
        model = CollisionStudentV5(DummyDecoder())
        clip = torch.rand(2, 16, 3, 64, 64)
        # full 256 is slow; use backbone F path with dummy tensors
        f = torch.randn(2, 128, 16, 16, requires_grad=True)
        z16 = torch.randn(2, 1, 16, 16, requires_grad=True)
        logits5 = torch.randn(2, 5, 5, requires_grad=True)
        out = {
            "F_pred": f,
            "Z_student_16": z16,
            "Hhat_1s_logits": logits5,
        }
        batch = {
            "F_teacher": torch.randn(2, 128, 16, 16),
            "Z_teacher_16": torch.randn(2, 1, 16, 16),
            "P_teacher_5": torch.rand(2, 5, 5),
        }
        loss, terms = student_v5_loss(out, batch, w_feat=0.5)
        loss.backward()
        self.assertTrue(f.grad.abs().sum() > 0)
        self.assertTrue(z16.grad.abs().sum() > 0)
        self.assertTrue(logits5.grad.abs().sum() > 0)
        self.assertIn("L_5x5_nbr", terms)
        del clip, model

    def test_gate_in_forward(self):
        model = CollisionStudentV5(DummyDecoder()).eval()
        clip = torch.rand(1, 16, 3, 256, 256)
        with torch.no_grad():
            out = model(clip)
        self.assertEqual(tuple(out["G_t"].shape), (1, 96, 8, 16, 16))
        self.assertEqual(tuple(out["F_pred"].shape), (1, 128, 16, 16))


if __name__ == "__main__":
    unittest.main()
