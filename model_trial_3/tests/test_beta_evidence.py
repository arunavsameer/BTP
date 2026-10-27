from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.losses.beta_evidence import (
    beta_nll,
    cell_stop_probability,
    evidential_mistake_penalty,
    noisy_or_stop,
    precision_from_logits,
    prob_safe_hinge,
    prob_stop_hinge,
)
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.checkpoint import extract_teacher_state, load_teacher_state


class BetaEvidenceTests(unittest.TestCase):
    def test_precision_at_least_one(self) -> None:
        phi = precision_from_logits(torch.tensor([-20.0, 0.0, 8.0]))
        self.assertGreaterEqual(float(phi.min()), 1.0)
        self.assertLessEqual(float(phi.max()), 64.0)

    def test_nll_lower_when_mean_matches(self) -> None:
        y = torch.full((1, 5, 5), 0.8)
        raw = torch.zeros(1, 5, 5)
        good = beta_nll(torch.full((1, 5, 5), 0.8), raw, y)
        bad = beta_nll(torch.full((1, 5, 5), 0.1), raw, y)
        self.assertLess(float(good), float(bad))
        self.assertTrue(torch.isfinite(good))

    def test_nll_prefers_high_precision_when_right(self) -> None:
        y = torch.full((1, 5, 5), 0.7)
        mu = torch.full((1, 5, 5), 0.7)
        sharp = beta_nll(mu, torch.full((1, 5, 5), 6.0), y)
        vague = beta_nll(mu, torch.full((1, 5, 5), -4.0), y)
        self.assertLess(float(sharp), float(vague))

    def test_evidential_penalizes_confident_mistakes(self) -> None:
        y = torch.zeros(1, 5, 5)
        mu = torch.ones(1, 5, 5)
        confident = evidential_mistake_penalty(mu, torch.full((1, 5, 5), 8.0), y)
        unsure = evidential_mistake_penalty(mu, torch.full((1, 5, 5), -8.0), y)
        self.assertGreater(float(confident), float(unsure))

    def test_cell_stop_prob_high_when_mean_is_stop(self) -> None:
        mu = torch.full((1, 5, 5), 0.90)
        raw = torch.full((1, 5, 5), 6.0)
        p = cell_stop_probability(mu, raw, threshold=0.60)
        self.assertGreater(float(p.min()), 0.9)

    def test_noisy_or_empty_is_near_zero_when_confident_safe(self) -> None:
        mu = torch.full((2, 5, 5), 0.02)
        raw = torch.full((2, 5, 5), 8.0)
        p = noisy_or_stop(mu, raw, threshold=0.60)
        self.assertEqual(tuple(p.shape), (2,))
        self.assertLess(float(p.max()), 0.05)

    def test_noisy_or_glow_is_not_safe(self) -> None:
        # Many 0.20 cells with low precision: tails leak past 0.60.
        glow = torch.full((1, 5, 5), 0.20)
        vague = noisy_or_stop(glow, torch.full((1, 5, 5), -6.0), threshold=0.60)
        sharp = noisy_or_stop(glow, torch.full((1, 5, 5), 8.0), threshold=0.60)
        self.assertGreater(float(vague), float(sharp))
        self.assertGreater(float(vague), 0.3)

    def test_prob_stop_hinge_fires_on_missed_stop(self) -> None:
        tgt = torch.zeros(1, 5, 5)
        tgt[0, 2, 2] = 0.9
        mu = torch.zeros(1, 5, 5)
        raw = torch.zeros(1, 5, 5)
        loss = prob_stop_hinge(mu, raw, tgt)
        self.assertGreater(float(loss), 0.0)
        mu2 = torch.zeros(1, 5, 5)
        mu2[0, 2, 2] = 0.95
        raw2 = torch.full((1, 5, 5), 6.0)
        self.assertLess(float(prob_stop_hinge(mu2, raw2, tgt)), 1e-5)

    def test_prob_safe_hinge_fires_on_false_stop(self) -> None:
        tgt = torch.zeros(1, 5, 5)
        mu = torch.zeros(1, 5, 5)
        mu[0, 2, 2] = 0.95
        raw = torch.full((1, 5, 5), 6.0)
        self.assertGreater(float(prob_safe_hinge(mu, raw, tgt)), 0.0)
        self.assertEqual(float(prob_safe_hinge(torch.zeros(1, 5, 5), raw, tgt)), 0.0)

    def test_composite_adds_beta_terms(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("loss.beta_nll_weight", 0.4)
        cfg.set("loss.evidential_weight", 0.05)
        cfg.set("loss.prob_stop_weight", 0.5)
        cfg.set("loss.prob_safe_weight", 0.5)
        loss_fn = TeacherLoss(cfg)
        pred = torch.rand(2, 5, 5, requires_grad=True)
        raw = torch.zeros(2, 5, 5, requires_grad=True)
        packed = loss_fn(
            {
                "Hhat_1s": pred,
                "Hhat_precision_logits": raw,
                "Z_t": torch.zeros(2, 32, 5, 5),
            },
            {"h_1s": torch.rand(2, 5, 5), "amb_1s": torch.zeros(2, dtype=torch.bool)},
        )
        packed.total.backward()
        self.assertTrue(torch.isfinite(packed.total))
        for k in ("L_beta", "L_evidential", "L_prob_stop", "L_prob_safe"):
            self.assertIn(k, packed.terms)
            self.assertTrue(torch.isfinite(packed.terms[k]))
        self.assertIsNotNone(pred.grad)
        self.assertIsNotNone(raw.grad)

    def test_evidence_head_roundtrip(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        torch.manual_seed(0)
        m = CollisionTeacher3(cfg, backbone=DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16))
        x = torch.rand(1, 16, 3, 64, 64)
        with torch.no_grad():
            a = m(x)["Hhat_precision_logits"].clone()
        m2 = CollisionTeacher3(cfg, backbone=DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16))
        load_teacher_state(m2, extract_teacher_state(m))
        m2.backbone.load_state_dict(m.backbone.state_dict())
        with torch.no_grad():
            b = m2(x)["Hhat_precision_logits"]
        self.assertTrue(torch.allclose(a, b, atol=1e-5))

    def test_evidence_head_gets_grad(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("loss.beta_nll_weight", 1.0)
        m = CollisionTeacher3(cfg, backbone=DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16))
        out = m(torch.rand(1, 16, 3, 64, 64))
        TeacherLoss(cfg)(
            out, {"h_1s": torch.rand(1, 5, 5), "amb_1s": torch.zeros(1, dtype=torch.bool)}
        ).total.backward()
        self.assertIsNotNone(m.evidence_head.weight.grad)

    def test_normal_cdf_matches_erfc(self) -> None:
        # P(H>=0.5 | μ=0.5) is 0.5 regardless of σ.
        p = cell_stop_probability(torch.full((1, 1, 1), 0.5), torch.zeros(1, 1, 1), threshold=0.5)
        self.assertAlmostEqual(float(p), 0.5, places=5)
        self.assertTrue(math.isfinite(float(p)))


if __name__ == "__main__":
    unittest.main()
