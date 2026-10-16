from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.losses.heatmap import false_positive_penalty, target_weighted_mse


class LossTests(unittest.TestCase):
    def test_weighted_mse_finite(self) -> None:
        pred = torch.rand(3, 5, 5)
        tgt = torch.rand(3, 5, 5)
        loss = target_weighted_mse(pred, tgt)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(loss.shape), ())

    def test_mask_excludes_samples(self) -> None:
        pred = torch.zeros(2, 5, 5)
        tgt = torch.ones(2, 5, 5)
        mask = torch.tensor([True, False])
        a = target_weighted_mse(pred, tgt, sample_mask=mask)
        b = target_weighted_mse(pred[:1], tgt[:1])
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_all_masked_zero(self) -> None:
        pred = torch.rand(2, 5, 5)
        tgt = torch.rand(2, 5, 5)
        mask = torch.zeros(2, dtype=torch.bool)
        loss = target_weighted_mse(pred, tgt, sample_mask=mask)
        self.assertEqual(float(loss), 0.0)

    def test_fp_no_safe_cells(self) -> None:
        pred = torch.ones(1, 5, 5)
        tgt = torch.ones(1, 5, 5)
        self.assertEqual(float(false_positive_penalty(pred, tgt)), 0.0)

    def test_composite_terms(self) -> None:
        cfg = Config.load()
        loss_fn = TeacherLoss(cfg)
        out = {
            "Hhat_now": torch.rand(2, 5, 5, requires_grad=True),
            "Hhat_1s": torch.rand(2, 5, 5, requires_grad=True),
            "Hhat_1_5s": torch.rand(2, 5, 5, requires_grad=True),
            "Hhat_2s": torch.rand(2, 5, 5, requires_grad=True),
            "Delta_1s": torch.zeros(2, 5, 5, requires_grad=True),
            "Delta_1_5s": torch.zeros(2, 5, 5, requires_grad=True),
            "Delta_2s": torch.zeros(2, 5, 5, requires_grad=True),
            "Z_t": torch.randn(2, 32, 5, 5, requires_grad=True),
        }
        batch = {
            "h_now": torch.rand(2, 5, 5),
            "h_1s": torch.rand(2, 5, 5),
            "h_1_5s": torch.rand(2, 5, 5),
            "h_2s": torch.rand(2, 5, 5),
            "amb_1s": torch.zeros(2, dtype=torch.bool),
            "amb_1_5s": torch.tensor([False, True]),
            "amb_2s": torch.zeros(2, dtype=torch.bool),
        }
        packed = loss_fn(out, batch)
        packed.total.backward()
        self.assertTrue(torch.isfinite(packed.total))
        for key in ("L_total", "L_future", "L_1s", "L_1_5s", "L_2s", "L_now", "L_delta", "L_FP", "L_dir", "L_bg"):
            self.assertIn(key, packed.terms)

    def test_recall_beta_one_matches_default(self) -> None:
        pred = torch.rand(2, 5, 5)
        tgt = torch.rand(2, 5, 5)
        a = target_weighted_mse(pred, tgt)
        b = target_weighted_mse(pred, tgt, recall_beta=1.0)
        self.assertAlmostEqual(float(a), float(b), places=6)

    def test_recall_beta_raises_underpredicted_high_cells(self) -> None:
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 4, 2] = 1.0
        pred = torch.zeros(1, 5, 5)
        pred[0, 4, 2] = 0.2
        pred[0, 0, 0] = 0.0
        a = target_weighted_mse(pred, tgt, recall_beta=1.0)
        b = target_weighted_mse(pred, tgt, recall_beta=8.0)
        self.assertGreater(float(b), float(a))


if __name__ == "__main__":
    unittest.main()
