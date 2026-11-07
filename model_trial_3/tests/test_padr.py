from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.losses.composite import TeacherLossOutput
from collision_jepa3.losses.heatmap import balanced_bce_logits, safe_logit_hinge, stop_logit_hinge
from collision_jepa3.training.pair_signals import (
    bounded_pair_perturbation,
    occupancy_balanced_huber,
    pair_gt_consistent,
    pair_padr_loss,
)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class TinyDecoder(nn.Module):
    def _heads(self, z, fine):
        pooled = F.adaptive_avg_pool2d(fine, 5).mean(dim=1)
        return {"Hhat_1s_logits": pooled}


class PadrTests(unittest.TestCase):
    def test_occupancy_huber_empty_uses_empty_term(self) -> None:
        za = torch.zeros(2, 5, 5)
        zb = torch.ones(2, 5, 5)
        y = torch.zeros(2, 5, 5)
        loss = occupancy_balanced_huber(za, zb, y)
        empty_only = F.smooth_l1_loss(za, zb, reduction="mean")
        self.assertAlmostEqual(float(loss), float(empty_only), places=5)

    def test_occupancy_huber_balances_occupied_and_empty(self) -> None:
        za = torch.zeros(1, 2, 2)
        zb = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        y = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
        loss = occupancy_balanced_huber(za, zb, y)
        pos = F.smooth_l1_loss(za[:, 0, 0], zb[:, 0, 0], reduction="mean")
        neg = F.smooth_l1_loss(za[:, 0, 1], zb[:, 0, 1], reduction="mean")
        self.assertAlmostEqual(float(loss), 0.5 * (float(pos) + float(neg)), places=5)

    def test_perturbation_rms_is_capped(self) -> None:
        fa = torch.ones(3, 4, 2, 2)
        fb = fa + 10.0
        fa_p, fb_p, rho, scale = bounded_pair_perturbation(fa, fb, rho_lo=0.09, rho_hi=0.10, rms_frac=0.05)
        pert = fa_p - fa
        rms_p = pert.flatten(1).pow(2).mean(1).sqrt()
        rms_f = 0.5 * (fa.flatten(1).pow(2).mean(1).sqrt() + fb.flatten(1).pow(2).mean(1).sqrt())
        self.assertTrue(torch.all(rms_p <= 0.05 * rms_f + 1e-5))
        self.assertTrue(torch.allclose(fb_p, fb - pert))
        self.assertTrue(torch.all(scale < 1.0))
        self.assertTrue(torch.all((rho >= 0.09) & (rho <= 0.10)))

    def test_delta_is_stopgrad(self) -> None:
        fa = torch.randn(2, 3, 4, 4, requires_grad=True)
        fb = torch.randn(2, 3, 4, 4, requires_grad=True)
        fa_p, fb_p, _, _ = bounded_pair_perturbation(fa, fb, rho_lo=0.05, rho_hi=0.05, rms_frac=1.0)
        fa_p.sum().backward()
        self.assertIsNotNone(fa.grad)
        self.assertTrue(fb.grad is None or torch.all(fb.grad == 0))

    def test_stop_logit_hinge_only_on_gt_stop(self) -> None:
        logits = torch.full((2, 5, 5), -2.0)
        tgt = torch.zeros(2, 5, 5)
        tgt[0, 2, 2] = 0.9
        loss, active, n = stop_logit_hinge(logits, tgt, stop_threshold=0.60, margin=0.1)
        tau = _logit(0.60)
        expected = tau + 0.1 - (-2.0)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(float(loss), expected, places=4)
        self.assertEqual(active, 1.0)
        zero, _, n_safe = stop_logit_hinge(logits[1:], tgt[1:], stop_threshold=0.60, margin=0.1)
        self.assertEqual(n_safe, 0)
        self.assertEqual(float(zero), 0.0)

    def test_safe_logit_hinge_only_on_gt_safe(self) -> None:
        logits = torch.full((2, 5, 5), 2.0)
        tgt = torch.zeros(2, 5, 5)
        tgt[0, 2, 2] = 0.9
        loss, active, n = safe_logit_hinge(logits, tgt, caution_threshold=0.25, margin=0.1)
        tau = _logit(0.25)
        expected = 2.0 - tau + 0.1
        self.assertEqual(n, 1)
        self.assertAlmostEqual(float(loss), expected, places=4)
        self.assertEqual(active, 1.0)

    def test_balanced_bce_empty_map_uses_empty_term(self) -> None:
        logits = torch.zeros(1, 5, 5)
        target = torch.zeros(1, 5, 5)
        got = balanced_bce_logits(logits, target)
        empty = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
        self.assertAlmostEqual(float(got), float(empty), places=5)

    def test_as_floats_keeps_optimizer_total(self) -> None:
        task = torch.tensor(0.2)
        extra = torch.tensor(0.05)
        packed = TeacherLossOutput(total=task, terms={"L_1s": task, "L_total": task})
        packed.total = packed.total + extra
        packed.terms["L_pair"] = extra
        floats = packed.as_floats()
        self.assertAlmostEqual(floats["L_total"], 0.25, places=5)
        self.assertAlmostEqual(floats["L_pair"], 0.05, places=5)

    def test_padr_tangent_tracks_endpoint_not_gt(self) -> None:
        model = TinyDecoder()
        fa = torch.zeros(2, 4, 8, 8)
        fb = torch.ones(2, 4, 8, 8)
        y = torch.ones(2, 5, 5)
        mask = torch.zeros(2, dtype=torch.bool)
        za = model._heads(None, fa)["Hhat_1s_logits"]
        lp, lt, stats = pair_padr_loss(
            model,
            z_a=torch.zeros(2, 1),
            fine_a=fa,
            z_b=torch.zeros(2, 1),
            fine_b=fb,
            y=y,
            mask=mask,
            za=za,
            rho_lo=0.05,
            rho_hi=0.05,
            rms_frac=0.05,
        )
        self.assertTrue(torch.isfinite(lp) and torch.isfinite(lt))
        self.assertGreater(float(lp), 0.0)
        self.assertGreaterEqual(float(lt), 0.0)
        self.assertEqual(stats["padr_n"], 2.0)

    def test_pair_gt_consistent_mask(self) -> None:
        gt_a = torch.zeros(3, 2, 2)
        gt_b = gt_a.clone()
        gt_b[1, 0, 0] = 0.5
        valid = pair_gt_consistent(gt_a, gt_b, tol=1e-3)
        self.assertEqual(valid.tolist(), [True, False, True])

    def test_padr_skips_gt_mismatched_pairs(self) -> None:
        model = TinyDecoder()
        fa = torch.zeros(2, 4, 8, 8)
        fb = torch.ones(2, 4, 8, 8)
        y = torch.ones(2, 5, 5)
        mask = torch.zeros(2, dtype=torch.bool)
        za = model._heads(None, fa)["Hhat_1s_logits"]
        pair_valid = torch.tensor([True, False])
        lp, lt, stats = pair_padr_loss(
            model,
            z_a=torch.zeros(2, 1),
            fine_a=fa,
            z_b=torch.zeros(2, 1),
            fine_b=fb,
            y=y,
            mask=mask,
            za=za,
            pair_valid=pair_valid,
            rho_lo=0.05,
            rho_hi=0.05,
            rms_frac=0.05,
        )
        self.assertEqual(stats["padr_n_gt_valid"], 1.0)
        self.assertEqual(stats["padr_n_gt_skip"], 1.0)
        self.assertEqual(stats["padr_n"], 2.0)
        self.assertTrue(torch.isfinite(lp))

    def test_padr_all_mismatch_zero_pair_loss(self) -> None:
        model = TinyDecoder()
        fa = torch.zeros(2, 4, 8, 8)
        fb = torch.ones(2, 4, 8, 8)
        y = torch.ones(2, 5, 5)
        mask = torch.zeros(2, dtype=torch.bool)
        za = model._heads(None, fa)["Hhat_1s_logits"]
        pair_valid = torch.zeros(2, dtype=torch.bool)
        lp, lt, stats = pair_padr_loss(
            model,
            z_a=torch.zeros(2, 1),
            fine_a=fa,
            z_b=torch.zeros(2, 1),
            fine_b=fb,
            y=y,
            mask=mask,
            za=za,
            pair_valid=pair_valid,
        )
        self.assertEqual(float(lp), 0.0)
        self.assertEqual(float(lt), 0.0)
        self.assertEqual(stats["padr_n_gt_valid"], 0.0)
        self.assertEqual(stats["padr_n_gt_skip"], 2.0)


if __name__ == "__main__":
    unittest.main()
