"""Shape / frozen-decoder / G-heatmap loss checks for the Trial-4 student."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa4.config import Config
from collision_jepa4.losses import TeacherLoss
from student.losses import student_loss
from student.model import StudentLatPairCF
from student.parent import decoder_state_hash, load_parent_decoder
from student.paths import EXPECTED_DECODER_HASH, parent_ckpt


def _dummy_out(batch=2):
    h = torch.rand(batch, 5, 5)
    logits = torch.logit(h.clamp(1e-4, 1 - 1e-4))
    return {
        "F": torch.randn(batch, 128, 16, 16, requires_grad=True),
        "Hhat_1s": h,
        "Hhat_1s_logits": logits,
    }


class StudentTests(unittest.TestCase):
    def test_parent_decoder_hash_and_shapes(self):
        ckpt = parent_ckpt()
        if not ckpt.is_file():
            self.skipTest(f"parent ckpt missing: {ckpt}")
        decoder = load_parent_decoder(ckpt)
        self.assertEqual(decoder_state_hash(decoder), EXPECTED_DECODER_HASH)
        model = StudentLatPairCF(decoder)
        audit = model.param_audit()
        self.assertLess(audit["total"], 1_000_000)
        self.assertEqual(audit["decoder"], 47809)
        self.assertFalse(any(p.requires_grad for p in model.decoder.parameters()))
        clip = torch.rand(1, 16, 3, 256, 256)
        out = model(clip)
        self.assertEqual(tuple(out["F"].shape), (1, 128, 16, 16))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (1, 5, 5))
        out["F"].sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))

    def test_student_loss_uses_fixed_weights(self):
        out = _dummy_out(2)
        h1s = torch.zeros(2, 5, 5)
        h1s[0, 4, 2] = 1.0
        batch = {
            "F_teacher": torch.randn(2, 128, 16, 16),
            "gt_1s": h1s,
            "h_1s": h1s,
            "h_now": torch.zeros(2, 5, 5),
            "amb_1s": torch.zeros(2, dtype=torch.bool),
        }
        mu = batch["F_teacher"].mean(dim=(0, 2, 3), keepdim=True)
        sigma = batch["F_teacher"].std(dim=(0, 2, 3), keepdim=True) + 1e-6
        feat_w, heatmap_w = 1.0, 2.0
        loss, terms = student_loss(
            out,
            batch,
            mu=mu,
            sigma=sigma,
            heatmap_loss=TeacherLoss(Config.load()),
            feat_weight=feat_w,
            heatmap_weight=heatmap_w,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(terms["feat_weight"], 1.0)
        self.assertAlmostEqual(terms["heatmap_weight"], 2.0)
        self.assertAlmostEqual(terms["L_feat_contrib"], terms["L_feat"] * feat_w, places=5)
        self.assertAlmostEqual(terms["L_heatmap_contrib"], terms["L_heatmap"] * heatmap_w, places=5)
        self.assertIn("L_1s", terms)
        self.assertIn("L_stop_hinge", terms)
        self.assertNotIn("L_cos", terms)
        self.assertNotIn("L_mgd", terms)
        loss.backward()
        self.assertIsNotNone(out["F"].grad)

    def test_g_appear_weight_uses_h_now(self):
        loss_fn = TeacherLoss(Config.load())
        h1s = torch.zeros(1, 5, 5)
        h1s[0, 4, 2] = 1.0
        logits = torch.full((1, 5, 5), -2.0)
        h = torch.sigmoid(logits)
        out = {"Hhat_1s": h, "Hhat_1s_logits": logits}
        a = loss_fn(out, {"h_1s": h1s, "h_now": torch.zeros(1, 5, 5)})
        b = loss_fn(out, {"h_1s": h1s, "h_now": torch.ones(1, 5, 5)})
        self.assertGreater(abs(float(a.terms["L_1s"]) - float(b.terms["L_1s"])), 1e-5)


if __name__ == "__main__":
    unittest.main()
