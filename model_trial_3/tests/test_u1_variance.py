from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.pair_variance import empirical_variance, envelope_max
from collision_jepa3.losses.composite import TeacherLoss
from collision_jepa3.evaluation.counterfactual import (
    feasible_vs_g,
    fit_mu_sigma_lookup,
    apply_mu_sigma_lookup,
    risk_maps,
    select_kappa,
    warning_report,
)
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.variance_head import VarianceHead
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.checkpoint import extract_teacher_state, load_teacher_state
from collision_jepa3.u1_runtime import weighted_var_loss


class PairVarianceMathTests(unittest.TestCase):
    def test_two_point_matches_closed_form(self) -> None:
        hc = np.array([[0.8, 0.1], [0.0, 1.0]], dtype=np.float32)
        hm = np.array([[0.1, 0.1], [0.0, 0.0]], dtype=np.float32)
        v = empirical_variance([hc, hm])
        expect = ((hc - hm) ** 2) / 4.0
        np.testing.assert_allclose(v, expect, atol=1e-6)
        env = envelope_max([hc, hm])
        np.testing.assert_allclose(env, np.maximum(hc, hm))
        mu = 0.5 * (hc + hm)
        sigma = np.sqrt(v)
        np.testing.assert_allclose(mu + sigma, np.maximum(hc, hm), atol=1e-6)

    def test_three_maps_uses_set_variance(self) -> None:
        maps = [
            np.zeros((2, 2), dtype=np.float32),
            np.ones((2, 2), dtype=np.float32),
            np.full((2, 2), 0.5, dtype=np.float32),
        ]
        v = empirical_variance(maps)
        self.assertAlmostEqual(float(v[0, 0]), float(np.var([0.0, 1.0, 0.5])), places=6)


class RiskScoreTests(unittest.TestCase):
    def test_kappa_zero_is_mu(self) -> None:
        mu = np.array([[[0.4, 0.1], [0.0, 0.8]]], dtype=np.float32)
        var = np.array([[[0.09, 0.0], [0.0, 0.0]]], dtype=np.float32)
        r0 = risk_maps(mu, var, 0.0)
        np.testing.assert_allclose(r0, mu)
        r1 = risk_maps(mu, var, 1.0)
        self.assertAlmostEqual(float(r1[0, 0, 0]), 0.7, places=5)
        self.assertLessEqual(float(r1.max()), 1.0)

    def test_warning_and_feasibility(self) -> None:
        n = 8
        gt = np.zeros((n, 5, 5), dtype=np.float32)
        pred = np.zeros((n, 5, 5), dtype=np.float32)
        gt[:4, 2, 2] = 0.9
        pred[:3, 2, 2] = 0.9
        pre = np.array([True] * 4 + [False] * 4)
        g = warning_report(pred, gt, pre_fork=pre)
        self.assertEqual(g["headline"]["stop_caught"], 3)
        cand = warning_report(pred, gt, pre_fork=pre)
        vs = feasible_vs_g(cand, g)
        self.assertTrue(vs["feasible"])


class HeadAndLossTests(unittest.TestCase):
    def test_head_range_and_param_count(self) -> None:
        h = VarianceHead(in_channels=128, out_grid=5)
        n = sum(p.numel() for p in h.parameters())
        self.assertEqual(n, 129)
        y = h(torch.zeros(2, 128, 16, 16)).detach()
        self.assertEqual(tuple(y.shape), (2, 5, 5))
        self.assertGreaterEqual(float(y.min()), 0.0)
        self.assertLessEqual(float(y.max()), 0.25)
        self.assertLess(float(y.mean()), 0.02)

    def test_composite_var_term(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("loss.var_weight", 1.0)
        loss_fn = TeacherLoss(cfg)
        v_true = torch.zeros(2, 5, 5)
        v_true[0, 0, 0] = 0.25
        out = {
            "Hhat_1s": torch.rand(2, 5, 5, requires_grad=True),
            "Hhat_var": torch.zeros(2, 5, 5, requires_grad=True),
            "Z_t": torch.randn(2, 32, 5, 5),
        }
        batch = {
            "h_1s": torch.rand(2, 5, 5),
            "amb_1s": torch.zeros(2, dtype=torch.bool),
            "v_1s": v_true,
            "v_mask": torch.tensor([True, True]),
        }
        zero = loss_fn(out, batch)
        out["Hhat_var"] = v_true.clone().requires_grad_(True)
        good = loss_fn(out, batch)
        self.assertIn("L_var", zero.terms)
        self.assertGreater(float(zero.terms["L_var"]), float(good.terms["L_var"]))

    def test_weighted_loss_does_not_ignore_positives(self) -> None:
        v_true = torch.zeros(2, 5, 5)
        v_true[0, 0, 0] = 0.25
        mask = torch.tensor([True, True])
        zero = weighted_var_loss(torch.zeros_like(v_true), v_true, mask, pos_weight=9.0)
        good = weighted_var_loss(v_true.clone(), v_true, mask, pos_weight=9.0)
        self.assertGreater(float(zero), float(good))

    def test_mu_sigma_lookup_tracks_inverted_u(self) -> None:
        rng = np.random.RandomState(0)
        mu = rng.rand(20, 5, 5).astype(np.float32)
        var = (mu * (1.0 - mu)).astype(np.float32)
        mask = np.ones(20, dtype=bool)
        look = fit_mu_sigma_lookup(mu, var, mask, n_bins=10)
        hat = apply_mu_sigma_lookup(mu, look)
        self.assertLess(float(np.mean(np.abs(hat - var))), 0.03)


class FrozenTeacherTests(unittest.TestCase):
    def _cfg(self, variance: bool) -> Config:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        cfg.set("adapter.variance_head", variance)
        return cfg

    def _model(self, variance: bool) -> CollisionTeacher3:
        cfg = self._cfg(variance)
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        return CollisionTeacher3(cfg, backbone=backbone)

    def test_default_direct_keys_unchanged(self) -> None:
        m = self._model(False)
        out = m(torch.rand(1, 16, 3, 64, 64))
        self.assertNotIn("Hhat_var", out)

    def test_variance_head_does_not_move_mu(self) -> None:
        m = self._model(True)
        clip = torch.rand(2, 16, 3, 64, 64)
        with torch.no_grad():
            mu0 = m(clip)["Hhat_1s"].clone()
        n = m.freeze_except_variance()
        self.assertEqual(n, 129)
        opt = torch.optim.SGD(m.variance_head.parameters(), lr=0.1)
        out = m(clip)
        loss = out["Hhat_var"].sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            mu1 = m(clip)["Hhat_1s"]
        self.assertTrue(torch.allclose(mu0, mu1, atol=1e-6))
        self.assertFalse(any(p.requires_grad for n, p in m.named_parameters() if "variance_head" not in n))

    def test_checkpoint_roundtrip(self) -> None:
        m = self._model(True)
        blob = extract_teacher_state(m)
        self.assertTrue(blob["variance_head"])
        m2 = self._model(True)
        load_teacher_state(m2, blob)
        self.assertTrue(
            torch.equal(m.variance_head.conv.bias.detach(), m2.variance_head.conv.bias.detach())
        )

    def test_select_kappa_rejects_unsafe(self) -> None:
        g = warning_report(
            np.zeros((4, 5, 5), dtype=np.float32),
            np.zeros((4, 5, 5), dtype=np.float32),
            pre_fork=np.array([True, True, False, False]),
        )
        bad = warning_report(
            np.ones((4, 5, 5), dtype=np.float32),
            np.zeros((4, 5, 5), dtype=np.float32),
            pre_fork=np.array([True, True, False, False]),
        )
        curves = [
            {"kappa": 0.0, "report": g, "vs_g": feasible_vs_g(g, g)},
            {"kappa": 1.0, "report": bad, "vs_g": feasible_vs_g(bad, g)},
        ]
        sel = select_kappa(curves, g)
        self.assertEqual(sel["kappa"], 0.0)


if __name__ == "__main__":
    unittest.main()
