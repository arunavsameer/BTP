"""Unit tests for the model-judge: prove old criteria fail and new ones don't."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.evaluation.judge import (
    HeatmapPack,
    borda_ranks,
    bootstrap_p_better,
    clip_aps,
    compare_models,
    gates_vs_prior,
    high_iou,
    mean_heatmap_prior,
    nested_recall_at_fa,
    peak_metrics,
    recall_at_fa_capped,
    score_pack,
    weighted_mse,
)
from collision_jepa3.evaluation.operating_point import recall_at_false_alarm


def _pack(pred, gt, clip_id=None, valid=None, **meta) -> HeatmapPack:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    n = pred.shape[0]
    if clip_id is None:
        clip_id = np.array([f"c{i}" for i in range(n)], dtype=object)
    return HeatmapPack(pred=pred, gt=gt, valid=valid if valid is not None else np.ones(n, dtype=bool), clip_id=clip_id, **meta)


def _localized(n=40, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One high cell per map, location varies by clip."""
    rng = np.random.RandomState(seed)
    gt = np.zeros((n, 5, 5), dtype=np.float64)
    clip = np.empty(n, dtype=object)
    for i in range(n):
        cid = i // 4
        clip[i] = f"clip{cid}"
        r, c = divmod((cid * 3 + (i % 4)) % 25, 5)
        gt[i, r, c] = 0.95
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 5 and 0 <= cc < 5:
                gt[i, rr, cc] = 0.35
    return gt, gt.copy(), clip


class CappedRecallTests(unittest.TestCase):
    def test_quantile_rule_claims_perfect_recall_on_a_constant_map(self) -> None:
        gt = np.zeros((20, 5, 5))
        gt[:, 2, 2] = 1.0
        pred = np.full_like(gt, 0.9)
        valid = np.ones_like(gt, dtype=bool)
        legacy = recall_at_false_alarm(pred, gt, valid, fa_rate=0.05)
        self.assertGreater(legacy["recall_at_fa"], 0.99)
        self.assertGreater(legacy["fa_achieved"], 0.9)

    def test_capped_recall_rejects_constant_map(self) -> None:
        gt = np.zeros((20, 5, 5))
        gt[:, 2, 2] = 1.0
        pred = np.full_like(gt, 0.9)
        capped = recall_at_fa_capped(pred, gt, fa_rate=0.05)
        self.assertLess(capped["recall"], 0.05)
        self.assertLessEqual(capped["fa"], 0.05 + 1e-9)

    def test_perfect_ranking_gets_full_capped_recall(self) -> None:
        gt = np.zeros((12, 5, 5))
        gt[:, 0, 0] = 1.0
        pred = gt.copy()
        capped = recall_at_fa_capped(pred, gt, fa_rate=0.05)
        self.assertGreater(capped["recall"], 0.99)
        self.assertLessEqual(capped["fa"], 0.05 + 1e-9)


class NestedVsInsampleTests(unittest.TestCase):
    def test_in_sample_quantile_is_optimistic_vs_nested(self) -> None:
        """Per-clip high negatives: in-sample 5% FA threshold leaks eval cells."""
        n_clips, n_win = 10, 8
        n = n_clips * n_win
        gt = np.zeros((n, 5, 5))
        pred = np.zeros((n, 5, 5))
        clip = np.empty(n, dtype=object)
        rng = np.random.RandomState(4)
        for i in range(n):
            cid = i // n_win
            clip[i] = f"c{cid}"
            gt[i, 0, 0] = 1.0
            pred[i, 0, 0] = 0.7
            # one extra loud false cell, unique per clip so in-sample 5% covers it
            r, c = 1 + (cid % 4), 1 + (i % 4)
            pred[i, r, c] = 0.85
            pred[i] += 0.02 * rng.rand(5, 5)
        pack = _pack(pred, gt, clip_id=clip)
        legacy = recall_at_false_alarm(pred, gt, np.ones_like(gt, dtype=bool), fa_rate=0.05)
        nested = nested_recall_at_fa(pack, fa_rate=0.05, n_folds=5, seed=0)
        self.assertGreaterEqual(legacy["recall_at_fa"] + 1e-9, nested["recall"])


class LocalizationTests(unittest.TestCase):
    def test_peak_shift_kills_hit_rate(self) -> None:
        gt, pred_ok, clip = _localized(40, seed=1)
        pred_shift = np.roll(gt, 1, axis=2)
        ok = _pack(pred_ok, gt, clip_id=clip)
        bad = _pack(pred_shift, gt, clip_id=clip)
        self.assertGreater(peak_metrics(ok)["peak_hit"], 0.99)
        self.assertLess(peak_metrics(bad)["peak_hit"], 0.05)
        self.assertGreater(high_iou(ok), high_iou(bad))

    def test_blur_loses_iou_to_a_copy_of_gt(self) -> None:
        gt, pred_ok, clip = _localized(32, seed=2)
        blur = 0.55 * np.ones_like(gt) + 0.2 * gt
        self.assertGreater(high_iou(_pack(pred_ok, gt, clip_id=clip)), high_iou(_pack(blur, gt, clip_id=clip)))


class ClipMacroTests(unittest.TestCase):
    def test_pooled_ap_can_hide_failure_on_most_clips(self) -> None:
        # One easy clip with many high cells dominates pooled AP; clip-macro does not.
        maps = []
        clips = []
        preds = []
        for w in range(8):
            g = np.zeros((5, 5))
            g[0:4, 0:4] = 1.0
            p = g.copy()
            maps.append(g)
            preds.append(p)
            clips.append("easy")
        for cid in range(10):
            for _ in range(2):
                g = np.zeros((5, 5))
                g[4, 4] = 1.0
                p = np.zeros((5, 5))
                p[0, 0] = 0.95
                maps.append(g)
                preds.append(p)
                clips.append(f"hard{cid}")
        pack = _pack(np.stack(preds), np.stack(maps), clip_id=np.asarray(clips, dtype=object))
        m = score_pack(pack, n_folds=4, seed=0)
        self.assertGreater(m["pooled_ap"], m["clip_macro_ap"] + 0.15)
        self.assertGreater(m["pooled_ap"], 0.5)
        self.assertLess(m["clip_macro_ap"], 0.35)


class CalibrationTests(unittest.TestCase):
    def test_loud_maps_win_mean_pred_gt80_but_lose_weighted_mse(self) -> None:
        gt, pred_ok, clip = _localized(36, seed=3)
        loud = np.clip(np.maximum(gt + 0.2, 0.85), 0.0, 1.0)
        ok = score_pack(_pack(pred_ok, gt, clip_id=clip), n_folds=4)
        bad = score_pack(_pack(loud, gt, clip_id=clip), n_folds=4)
        self.assertGreater(bad["mean_pred_gt80"], ok["mean_pred_gt80"])
        self.assertLess(ok["weighted_mse"], bad["weighted_mse"])
        self.assertLess(ok["high_mae"], bad["high_mae"])


class GateAndBordaTests(unittest.TestCase):
    def test_mean_prior_fails_gate_against_perfect_maps(self) -> None:
        gt, pred, clip = _localized(40, seed=5)
        pack = _pack(pred, gt, clip_id=clip)
        prior = mean_heatmap_prior(gt, np.ones(len(gt), dtype=bool), pack)
        sm = score_pack(pack, n_folds=4)
        sp = score_pack(prior, n_folds=4)
        g = gates_vs_prior(sm, sp)
        self.assertTrue(g["pass"])
        self.assertFalse(gates_vs_prior(sp, sp)["pass"])

    def test_borda_picks_localized_over_loud_even_if_legacy_rec5_is_close(self) -> None:
        gt, pred_ok, clip = _localized(48, seed=6)
        loud = np.clip(gt + 0.15, 0.0, 1.0)
        loud = np.maximum(loud, 0.12)
        packs = {
            "localized": _pack(pred_ok, gt, clip_id=clip),
            "loud": _pack(loud, gt, clip_id=clip),
            "mean_prior": mean_heatmap_prior(gt, np.ones(len(gt), dtype=bool), _pack(pred_ok, gt, clip_id=clip)),
        }
        report = compare_models(packs, n_folds=4, n_boot=40, seed=0)
        self.assertEqual(report["winner"], "localized")
        self.assertTrue(report["gates"]["localized"]["pass"])
        self.assertLess(report["borda_all"]["localized"], report["borda_all"]["loud"])

    def test_constant_map_is_not_eligible(self) -> None:
        gt, pred_ok, clip = _localized(32, seed=7)
        const = np.full_like(gt, 0.8)
        packs = {
            "good": _pack(pred_ok, gt, clip_id=clip),
            "constant": _pack(const, gt, clip_id=clip),
            "mean_prior": mean_heatmap_prior(gt, np.ones(len(gt), dtype=bool), _pack(pred_ok, gt, clip_id=clip)),
        }
        report = compare_models(packs, n_folds=4, n_boot=20, seed=1)
        self.assertNotIn("constant", report["eligible"])
        self.assertEqual(report["winner"], "good")


class BootstrapTests(unittest.TestCase):
    def test_identical_models_are_not_separated(self) -> None:
        gt, pred, clip = _localized(40, seed=8)
        a = _pack(pred, gt, clip_id=clip)
        b = _pack(pred + 1e-8, gt, clip_id=clip)

        def _ap(p: HeatmapPack) -> float:
            return float(clip_aps(p)["clip_macro_ap"])

        p_ab = bootstrap_p_better(a, b, _ap, n_boot=60, seed=0)
        p_ba = bootstrap_p_better(b, a, _ap, n_boot=60, seed=1)
        self.assertLess(p_ab, 0.9)
        self.assertLess(p_ba, 0.9)

    def test_clearly_better_model_wins_bootstrap(self) -> None:
        gt, pred, clip = _localized(40, seed=9)
        noise = np.random.RandomState(0).rand(*gt.shape) * 0.8
        a = _pack(pred, gt, clip_id=clip)
        b = _pack(noise, gt, clip_id=clip)

        def _ap(p: HeatmapPack) -> float:
            return float(clip_aps(p)["clip_macro_ap"])

        p = bootstrap_p_better(a, b, _ap, n_boot=60, seed=0)
        self.assertGreaterEqual(p, 0.9)


class WeightedMseTests(unittest.TestCase):
    def test_matching_peaks_beats_matching_zeros(self) -> None:
        gt = np.zeros((8, 5, 5))
        gt[:, 4, 2] = 1.0
        peak = gt.copy()
        zeros = np.zeros_like(gt)
        clip = np.array([f"c{i}" for i in range(8)], dtype=object)
        self.assertLess(weighted_mse(_pack(peak, gt, clip_id=clip)), weighted_mse(_pack(zeros, gt, clip_id=clip)))


class BordaTieTests(unittest.TestCase):
    def test_tied_metrics_share_rank(self) -> None:
        board = {
            "a": {"nested_rec_at_2fa": 0.5, "nested_rec_at_5fa": 0.5, "clip_macro_ap": 0.5, "peak_hit": 0.5, "high_iou": 0.5, "high_mae": 0.1},
            "b": {"nested_rec_at_2fa": 0.5, "nested_rec_at_5fa": 0.5, "clip_macro_ap": 0.5, "peak_hit": 0.5, "high_iou": 0.5, "high_mae": 0.1},
        }
        ranks = borda_ranks(board, ["a", "b"])
        self.assertAlmostEqual(ranks["a"], ranks["b"])


if __name__ == "__main__":
    unittest.main()
