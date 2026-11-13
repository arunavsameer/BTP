from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.checkpoint import extract_teacher_state

from student_model.loss import StudentLoss, blob_weights, spatial_cosine, spatial_l1, weighted_l1
from student_model.student import MAX_PARAMS, CollisionStudent, load_frozen_teacher_heads


def _cfg() -> Config:
    cfg = Config.load()
    cfg.set("heads.mode", "direct_h1s")
    cfg.set("loss.close_weight", 1.0)
    return cfg


def _dummy_teacher(cfg):
    backbone = DummyVJEPA2Backbone(
        hidden_size=1408, crop_size=256, patch_size=16, tubelet_size=2, num_frames=16, num_layers=1
    )
    return CollisionTeacher3(cfg, backbone=backbone)


class StudentTests(unittest.TestCase):
    def test_fast_encoder_is_smaller_same_shapes(self) -> None:
        cfg = _cfg()
        cfg.set("student.encoder_variant", "fast")
        m = CollisionStudent(cfg)
        full = CollisionStudent(_cfg())
        audit = m.parameter_audit()
        self.assertLess(audit["total"], MAX_PARAMS)
        self.assertLess(audit["encoder"], full.parameter_audit()["encoder"])
        clip = torch.rand(2, 16, 3, 256, 256)
        out = m(clip)
        inf = m.infer(clip)
        self.assertEqual(tuple(out["F"].shape), (2, 128, 16, 16))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (2, 5, 5))
        self.assertEqual(tuple(inf["F"].shape), (2, 128, 16, 16))

    def test_under_one_million_params(self) -> None:
        m = CollisionStudent(_cfg())
        audit = m.parameter_audit()
        self.assertLess(audit["total"], MAX_PARAMS)
        self.assertGreater(audit["trainable"], 0)
        self.assertGreater(audit["frozen"], 0)
        self.assertGreater(audit["predictor"], 0)
        self.assertTrue(audit["under_1m"])

    def test_shapes(self) -> None:
        m = CollisionStudent(_cfg())
        clip = torch.rand(2, 16, 3, 256, 256)
        out = m(clip)
        self.assertEqual(tuple(out["F"].shape), (2, 128, 16, 16))
        self.assertEqual(tuple(out["F_ctx"].shape), (2, 128, 16, 16))
        self.assertEqual(tuple(out["Z_t"].shape), (2, 32, 5, 5))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (2, 5, 5))

    def test_decoder_stays_frozen(self) -> None:
        cfg = _cfg()
        m = CollisionStudent(cfg)
        clip = torch.rand(1, 16, 3, 256, 256)
        out = m(clip)
        t_out = {"F": torch.randn_like(out["F"]), "Z_t": torch.randn_like(out["Z_t"])}
        StudentLoss(cfg)(out, {}, t_out, epoch=0, predictor=m.predictor).total.backward()
        self.assertIsNone(m.decoder.net[0].weight.grad)
        self.assertIsNone(m.bottleneck[0].weight.grad)
        self.assertTrue(any(p.grad is not None for p in m.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in m.predictor.parameters()))

    def test_predictor_identity_at_init(self) -> None:
        m = CollisionStudent(_cfg())
        fine = torch.randn(2, 128, 16, 16)
        with torch.no_grad():
            self.assertTrue(torch.allclose(m.predictor(fine), fine, atol=1e-5))

    def test_copied_decoder_matches_teacher_on_same_fine(self) -> None:
        cfg = _cfg()
        torch.manual_seed(0)
        teacher = _dummy_teacher(cfg)
        student = CollisionStudent(cfg)
        load_frozen_teacher_heads(student, extract_teacher_state(teacher))
        fine = torch.randn(2, 128, 16, 16)
        with torch.no_grad():
            h_t, _ = teacher.decoder.forward_with_logits(fine)
            h_s, _ = student.decoder.forward_with_logits(fine)
        self.assertTrue(torch.allclose(h_t, h_s, atol=1e-5))

    def test_loss_is_latent_only(self) -> None:
        cfg = _cfg()
        f = torch.randn(2, 128, 16, 16, requires_grad=True)
        s_out = {
            "F": f,
            "F_ctx": f.detach(),
            "Z_t": torch.randn(2, 32, 5, 5, requires_grad=True),
            "Hhat_1s": torch.rand(2, 5, 5, requires_grad=True),
        }
        t_out = {"F": torch.randn(2, 128, 16, 16), "Z_t": torch.randn(2, 32, 5, 5)}
        packed = StudentLoss(cfg)(s_out, {"h_1s": torch.ones(2, 5, 5)}, t_out, epoch=0)
        packed.total.backward()
        self.assertTrue(torch.isfinite(packed.total))
        self.assertIn("L_feat", packed.terms)
        self.assertNotIn("L_gt", packed.terms)
        self.assertNotIn("L_kd", packed.terms)

    def test_l1_zero_when_equal(self) -> None:
        a = torch.randn(2, 128, 16, 16)
        self.assertLess(float(spatial_l1(a, a)), 1e-6)
        self.assertLess(float(spatial_cosine(a, a)), 1e-5)
        self.assertGreater(float(spatial_l1(a * 0, a)), 0.0)

    def test_no_teacher_zero_loss(self) -> None:
        cfg = _cfg()
        s_out = {"F": torch.randn(1, 128, 16, 16, requires_grad=True)}
        packed = StudentLoss(cfg)(s_out, {}, None, epoch=0)
        self.assertLess(float(packed.total.abs()), 1e-8)

    def test_blob_weight_penalizes_appear_cells(self) -> None:
        f_t = torch.zeros(1, 4, 16, 16)
        f_t[:, :, 0, 0] = 3.0
        h_now = torch.zeros(1, 5, 5)
        h_1s = torch.zeros(1, 5, 5)
        h_1s[:, 0, 0] = 0.90
        w = blob_weights(f_t, h_now, h_1s, energy_alpha=4.0, appear_boost=4.0, stop_boost=2.0)
        self.assertGreater(float(w[0, 0, 0, 0]), float(w[0, 0, 8, 8]))
        pred = f_t.clone()
        pred[:, :, 0, 0] = 0.0
        pred[:, :, 8, 8] = 3.0
        miss_blob = weighted_l1(pred, f_t, w)
        pred2 = f_t.clone()
        pred2[:, :, 8, 8] = 3.0
        miss_bg = weighted_l1(pred2, f_t, w)
        self.assertGreater(float(miss_blob), float(miss_bg))

    def test_last_tubelet_skip_present(self) -> None:
        m = CollisionStudent(_cfg())
        self.assertTrue(hasattr(m.encoder, "last_gate"))
        self.assertEqual(int(m.encoder.last_gate.numel()), 1)
        self.assertEqual(float(m.encoder.last_gate.item()), 0.0)

    def test_teacher_heatmap_kd_keeps_decoder_frozen(self) -> None:
        cfg = _cfg()
        cfg.set("student.h_weight", 1.0)
        cfg.set("student.peak_stop_weight", 1.0)
        cfg.set("student.peak_safe_weight", 1.0)
        m = CollisionStudent(cfg)
        clip = torch.rand(2, 16, 3, 256, 256)
        out = m(clip)
        t_out = {
            "F": torch.randn_like(out["F"]),
            "Z_t": torch.randn_like(out["Z_t"]),
            "Hhat_1s": torch.ones_like(out["Hhat_1s"]),
        }
        packed = StudentLoss(cfg)(
            out, {"h_now": torch.zeros(2, 5, 5), "h_1s": torch.ones(2, 5, 5)}, t_out, epoch=0
        )
        packed.total.backward()
        self.assertGreater(float(packed.terms["L_stop"].detach()), 0.0)
        self.assertIsNone(m.decoder.net[0].weight.grad)
        self.assertTrue(any(p.grad is not None for p in m.encoder.parameters()))

    def test_cellwise_safe_hinge_on_background(self) -> None:
        cfg = _cfg()
        cfg.set("student.h_weight", 1.0)
        cfg.set("student.peak_stop_weight", 1.0)
        cfg.set("student.peak_safe_weight", 1.0)
        h_s = torch.full((1, 5, 5), 0.90, requires_grad=True)
        h_t = torch.zeros(1, 5, 5)
        h_t[0, 0, 0] = 0.90
        packed = StudentLoss(cfg)(
            {"F": torch.zeros(1, 128, 16, 16), "Hhat_1s": h_s},
            {},
            {"F": torch.zeros(1, 128, 16, 16), "Hhat_1s": h_t},
        )
        self.assertGreater(float(packed.terms["L_safe"].detach()), 0.0)
        packed.total.backward()
        self.assertIsNotNone(h_s.grad)


if __name__ == "__main__":
    unittest.main()
