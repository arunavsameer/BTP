from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.losses.heatmap import (
    appear_cell_weight,
    blob_bce,
    blob_bce_logits,
    false_positive_penalty,
    maxpool3x3,
    safe_peak_hinge,
    small_blob_cell_weight,
    soft_bce,
    soft_bce_logits,
    stop_peak_hinge,
    target_weighted_mse,
)


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
        for key in ("L_total", "L_future", "L_1s", "L_1_5s", "L_2s", "L_now", "L_delta", "L_blob", "L_FP", "L_dir", "L_bg"):
            self.assertIn(key, packed.terms)

    def test_soft_bce_finite(self) -> None:
        pred = torch.rand(3, 5, 5)
        tgt = torch.rand(3, 5, 5)
        loss = soft_bce(pred, tgt)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(loss.shape), ())

    def test_soft_bce_mask_excludes_samples(self) -> None:
        pred = torch.full((2, 5, 5), 0.2)
        tgt = torch.full((2, 5, 5), 0.8)
        mask = torch.tensor([True, False])
        a = soft_bce(pred, tgt, sample_mask=mask)
        b = soft_bce(pred[:1], tgt[:1])
        self.assertAlmostEqual(float(a), float(b), places=5)

    def test_soft_bce_all_masked_zero(self) -> None:
        pred = torch.rand(2, 5, 5)
        tgt = torch.rand(2, 5, 5)
        mask = torch.zeros(2, dtype=torch.bool)
        loss = soft_bce(pred, tgt, sample_mask=mask)
        self.assertEqual(float(loss), 0.0)

    def test_soft_bce_pos_weight_upweights_occupied(self) -> None:
        pred = torch.full((1, 5, 5), 0.01)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.8
        a = soft_bce(pred, tgt, pos_weight=1.0)
        b = soft_bce(pred, tgt, pos_weight=2.0)
        self.assertGreater(float(b), float(a))

    def test_soft_bce_calibrated_on_coin_flip(self) -> None:
        tgt = torch.full((1, 5, 5), 0.5)
        half = soft_bce(torch.full((1, 5, 5), 0.5), tgt)
        peaked = soft_bce(torch.full((1, 5, 5), 0.95), tgt)
        self.assertLess(float(half), float(peaked))

    def test_soft_bce_no_row_weights(self) -> None:
        pred = torch.zeros(1, 5, 5)
        tgt = torch.zeros(1, 5, 5)
        pred[0, 0, 2] = 0.05
        tgt[0, 0, 2] = 0.8
        far = soft_bce(pred, tgt)
        pred2 = torch.zeros(1, 5, 5)
        tgt2 = torch.zeros(1, 5, 5)
        pred2[0, 4, 2] = 0.05
        tgt2[0, 4, 2] = 0.8
        near = soft_bce(pred2, tgt2)
        self.assertAlmostEqual(float(far), float(near), places=5)

    def test_appear_cell_weight_only_empty_to_occupied(self) -> None:
        h_now = torch.zeros(1, 5, 5)
        h_now[0, 0, 0] = 0.8
        h_fut = torch.zeros(1, 5, 5)
        h_fut[0, 0, 0] = 0.9
        h_fut[0, 2, 2] = 0.7
        w = appear_cell_weight(h_now, h_fut, now_threshold=0.1, future_threshold=0.5, weight=8.0)
        self.assertEqual(float(w[0, 0, 0]), 1.0)
        self.assertEqual(float(w[0, 2, 2]), 8.0)
        self.assertEqual(float(w[0, 1, 1]), 1.0)

    def test_soft_bce_appear_weight_upweights_new_cells(self) -> None:
        pred = torch.full((1, 5, 5), 0.01)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.8
        h_now = torch.zeros(1, 5, 5)
        extra = appear_cell_weight(h_now, tgt, weight=8.0)
        a = soft_bce(pred, tgt, pos_weight=2.0)
        b = soft_bce(pred, tgt, pos_weight=2.0, cell_weight=extra)
        self.assertGreater(float(b), float(a))

    def test_composite_heatmap_is_bce(self) -> None:
        cfg = Config.load()
        self.assertEqual(cfg.get("loss.heatmap"), "bce")
        self.assertEqual(cfg.get("loss.bce_pos_threshold"), 0.1)
        self.assertEqual(cfg.get("loss.bce_pos_weight"), 2.0)
        self.assertEqual(cfg.get("loss.bce_appear_weight"), 8.0)
        self.assertEqual(cfg.get("loss.bce_appear_now_threshold"), 0.1)
        self.assertEqual(cfg.get("loss.bce_appear_future_threshold"), 0.5)
        self.assertEqual(cfg.get("loss.blob_bce_weight"), 0.35)
        self.assertEqual(cfg.get("loss.blob_occupied_threshold"), 0.5)
        cfg.set("heads.mode", "direct_h1s")
        loss_fn = TeacherLoss(cfg)
        pred = torch.rand(2, 5, 5, requires_grad=True)
        tgt = torch.rand(2, 5, 5)
        h_now = torch.rand(2, 5, 5)
        amb = torch.tensor([False, True])
        packed = loss_fn(
            {"Hhat_1s": pred, "Z_t": torch.randn(2, 32, 5, 5)},
            {"h_1s": tgt, "h_now": h_now, "amb_1s": amb},
        )
        extra = appear_cell_weight(h_now, tgt, now_threshold=0.1, future_threshold=0.5, weight=8.0)
        expected = soft_bce(
            pred, tgt, sample_mask=~amb, pos_threshold=0.1, pos_weight=2.0, cell_weight=extra
        )
        self.assertAlmostEqual(float(packed.terms["L_1s"].detach()), float(expected.detach()), places=5)

    def test_composite_appear_weight_raises_missed_new_cells(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        pred = torch.full((1, 5, 5), 0.01)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.9
        h_now = torch.zeros(1, 5, 5)
        amb = torch.zeros(1, dtype=torch.bool)
        batch = {"h_1s": tgt, "h_now": h_now, "amb_1s": amb}
        out = {"Hhat_1s": pred, "Z_t": torch.zeros(1, 32, 5, 5)}
        high = TeacherLoss(cfg)(out, batch)
        cfg.set("loss.bce_appear_weight", 1.0)
        low = TeacherLoss(cfg)(out, batch)
        self.assertGreater(float(high.terms["L_1s"]), float(low.terms["L_1s"]))

    def test_maxpool3x3_includes_center_and_neighbor(self) -> None:
        x = torch.zeros(1, 5, 5)
        x[0, 2, 2] = 0.8
        p = maxpool3x3(x)
        self.assertAlmostEqual(float(p[0, 2, 2]), 0.8)
        self.assertAlmostEqual(float(p[0, 2, 3]), 0.8)
        self.assertAlmostEqual(float(p[0, 0, 0]), 0.0)

    def test_blob_bce_zero_on_empty_neighbourhoods(self) -> None:
        pred = torch.full((1, 5, 5), 0.01)
        tgt = torch.zeros(1, 5, 5)
        self.assertEqual(float(blob_bce(pred, tgt)), 0.0)

    def test_blob_bce_neighbor_beats_all_dark(self) -> None:
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.9
        dark = torch.full((1, 5, 5), 0.01)
        neigh = dark.clone()
        neigh[0, 2, 3] = 0.9
        exact = dark.clone()
        exact[0, 2, 2] = 0.9
        self.assertGreater(float(blob_bce(dark, tgt)), float(blob_bce(neigh, tgt)))
        self.assertGreater(float(blob_bce(neigh, tgt)), float(blob_bce(exact, tgt)))

    def test_composite_blob_added_to_total_not_l1s(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        pred = torch.full((1, 5, 5), 0.01, requires_grad=True)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.9
        h_now = torch.zeros(1, 5, 5)
        amb = torch.zeros(1, dtype=torch.bool)
        packed = TeacherLoss(cfg)(
            {"Hhat_1s": pred, "Z_t": torch.zeros(1, 32, 5, 5)},
            {"h_1s": tgt, "h_now": h_now, "amb_1s": amb},
        )
        extra = appear_cell_weight(h_now, tgt, now_threshold=0.1, future_threshold=0.5, weight=8.0)
        cell = soft_bce(pred, tgt, sample_mask=~amb, pos_threshold=0.1, pos_weight=2.0, cell_weight=extra)
        blob = blob_bce(pred, tgt, sample_mask=~amb, occupied_threshold=0.5)
        self.assertAlmostEqual(float(packed.terms["L_1s"].detach()), float(cell.detach()), places=5)
        self.assertAlmostEqual(float(packed.terms["L_blob"].detach()), float(blob.detach()), places=5)
        packed.total.backward()
        self.assertIsNotNone(pred.grad)
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_soft_bce_logits_near_zero_has_grad(self) -> None:
        logits = torch.full((1, 5, 5), -40.0, requires_grad=True)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 1.0
        loss = soft_bce_logits(logits, tgt, pos_weight=2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertLess(float(logits.grad[0, 2, 2]), 0.0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_blob_bce_logits_near_zero_has_grad(self) -> None:
        logits = torch.full((1, 5, 5), -40.0, requires_grad=True)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.9
        loss = blob_bce_logits(logits, tgt)
        loss.backward()
        self.assertGreater(float(loss), 0.0)
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertLess(float(logits.grad[0, 2, 2]), 0.0)

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

    def test_stop_peak_hinge_only_on_gt_stop(self) -> None:
        pred = torch.full((2, 5, 5), 0.40)
        tgt = torch.zeros(2, 5, 5)
        tgt[0, 2, 2] = 0.80
        tgt[1, 2, 2] = 0.20
        loss = stop_peak_hinge(pred, tgt, stop_threshold=0.60, margin=0.05)
        self.assertAlmostEqual(float(loss), 0.25, places=5)
        pred_ok = pred.clone()
        pred_ok[0, 2, 2] = 0.70
        self.assertEqual(float(stop_peak_hinge(pred_ok, tgt, stop_threshold=0.60, margin=0.05)), 0.0)
        amb = torch.tensor([True, False])
        self.assertEqual(float(stop_peak_hinge(pred, tgt, sample_mask=~amb)), 0.0)

    def test_stop_peak_hinge_has_grad_on_peak_cell(self) -> None:
        base = torch.zeros(1, 5, 5)
        base[0, 1, 1] = 0.20
        pred = base.clone().requires_grad_(True)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 1, 1] = 0.90
        stop_peak_hinge(pred, tgt).backward()
        self.assertIsNotNone(pred.grad)
        self.assertLess(float(pred.grad[0, 1, 1]), 0.0)

    def test_small_blob_upweights_two_cell_stop_not_large(self) -> None:
        small = torch.zeros(1, 5, 5)
        small[0, 2, 2] = 0.80
        small[0, 2, 3] = 0.70
        w_small = small_blob_cell_weight(small, weight=4.0, max_cells=4)
        self.assertEqual(float(w_small[0, 2, 2]), 4.0)
        self.assertEqual(float(w_small[0, 0, 0]), 1.0)
        large = torch.zeros(1, 5, 5)
        large[0, :2, :4] = 0.90
        w_large = small_blob_cell_weight(large, weight=4.0, max_cells=4)
        self.assertTrue(torch.equal(w_large, torch.ones_like(large)))

    def test_composite_hinge_and_small_blob_raise_missed_small_stop(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("loss.stop_hinge_weight", 0.0)
        cfg.set("loss.small_blob_weight", 1.0)
        pred = torch.full((1, 5, 5), 0.20, requires_grad=True)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.80
        h_now = tgt.clone()
        amb = torch.zeros(1, dtype=torch.bool)
        batch = {"h_1s": tgt, "h_now": h_now, "amb_1s": amb}
        out = {"Hhat_1s": pred, "Z_t": torch.zeros(1, 32, 5, 5)}
        base = TeacherLoss(cfg)(out, batch)
        cfg.set("loss.stop_hinge_weight", 1.0)
        cfg.set("loss.small_blob_weight", 4.0)
        boosted = TeacherLoss(cfg)(out, batch)
        self.assertGreater(float(boosted.total), float(base.total))
        self.assertGreater(float(boosted.terms["L_stop_hinge"]), 0.0)
        boosted.total.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_safe_peak_hinge_only_on_gt_safe(self) -> None:
        pred = torch.full((2, 5, 5), 0.50)
        tgt = torch.zeros(2, 5, 5)
        tgt[0, 2, 2] = 0.05
        tgt[1, 2, 2] = 0.80
        loss = safe_peak_hinge(pred, tgt, caution_threshold=0.25, margin=0.0)
        self.assertAlmostEqual(float(loss), 0.25, places=5)
        quiet = pred.clone()
        quiet[0] = 0.10
        self.assertEqual(float(safe_peak_hinge(quiet, tgt, caution_threshold=0.25)), 0.0)

    def test_safe_hinge_does_not_touch_stop_maps(self) -> None:
        pred = torch.full((1, 5, 5), 0.90)
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.80
        self.assertEqual(float(safe_peak_hinge(pred, tgt)), 0.0)

    def test_composite_safe_hinge_raises_false_stop(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("loss.stop_hinge_weight", 0.0)
        cfg.set("loss.safe_hinge_weight", 0.0)
        cfg.set("loss.small_blob_weight", 1.0)
        pred = torch.full((1, 5, 5), 0.70, requires_grad=True)
        tgt = torch.zeros(1, 5, 5)
        batch = {"h_1s": tgt, "h_now": tgt, "amb_1s": torch.zeros(1, dtype=torch.bool)}
        out = {"Hhat_1s": pred, "Z_t": torch.zeros(1, 32, 5, 5)}
        base = TeacherLoss(cfg)(out, batch)
        cfg.set("loss.safe_hinge_weight", 1.0)
        quiet = TeacherLoss(cfg)(out, batch)
        self.assertGreater(float(quiet.total), float(base.total))
        self.assertGreater(float(quiet.terms["L_safe_hinge"]), 0.0)
        quiet.total.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())


if __name__ == "__main__":
    unittest.main()
