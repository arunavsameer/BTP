from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.models.task_whiten import TaskWhitenDrop, drop_matrix, fit_drop_matrix
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.checkpoint import extract_teacher_state, load_teacher_state


class TaskWhitenTests(unittest.TestCase):
    def test_disabled_is_identity(self) -> None:
        tw = TaskWhitenDrop(channels=8)
        x = torch.randn(2, 8, 4, 4)
        self.assertTrue(torch.equal(tw(x), x))

    def test_drop_first_channel(self) -> None:
        tw = TaskWhitenDrop(channels=4)
        t = torch.eye(4)
        v = torch.zeros(4, 1)
        v[0, 0] = 1.0
        tw.load_W(drop_matrix(t, t, v))
        x = torch.ones(1, 4, 2, 2)
        y = tw(x)
        self.assertTrue(torch.allclose(y[:, 0], torch.zeros(1, 2, 2)))
        self.assertTrue(torch.allclose(y[:, 1], torch.ones(1, 2, 2)))

    def test_fit_drops_pair_only_direction(self) -> None:
        n, c, h, w = 64, 8, 4, 4
        fine = torch.zeros(n, c, h, w)
        # even/odd are a matched pair: channel 7 is background, channel 0 tracks occupancy
        gt = torch.zeros(n, 5, 5)
        for i in range(0, n, 2):
            occ = float((i // 2) % 2)
            fine[i, 0] = occ
            fine[i + 1, 0] = occ
            fine[i, 7] = 1.0
            fine[i + 1, 7] = -1.0
            gt[i] = occ
            gt[i + 1] = occ
        mate = torch.arange(n)
        mate[0::2] = torch.arange(1, n, 2)
        mate[1::2] = torch.arange(0, n, 2)
        stats = fit_drop_matrix(fine, gt, mate, drop_k=1)
        tw = TaskWhitenDrop(channels=c)
        tw.load_W(stats["W"])
        y = tw(fine)
        self.assertLess(float(y[:, 7].abs().mean()), float(fine[:, 7].abs().mean()) * 0.4)
        self.assertGreater(float(y[:, 0].abs().mean()), 0.4)

    def test_checkpoint_roundtrip_keeps_W(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        w = torch.eye(128)
        w[0, 0] = 0.0
        m.task_whiten.load_W(w)
        blob = extract_teacher_state(m)
        m2 = CollisionTeacher3(cfg, backbone=DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16))
        load_teacher_state(m2, blob)
        self.assertEqual(int(m2.task_whiten.enabled.item()), 1)
        self.assertTrue(torch.allclose(m2.task_whiten.W, w))


if __name__ == "__main__":
    unittest.main()
