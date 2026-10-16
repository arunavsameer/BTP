from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.checkpoint import extract_teacher_state, load_teacher_state, save_checkpoint, load_checkpoint
from collision_jepa3.training.optimizer import build_param_groups


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_predictions(self) -> None:
        cfg = Config.load()
        torch.manual_seed(0)
        b = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=b)
        x = torch.rand(1, 16, 3, 64, 64)
        with torch.no_grad():
            a = m(x)["Hhat_1s"].clone()
        blob = extract_teacher_state(m)
        torch.manual_seed(1)
        b2 = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m2 = CollisionTeacher3(cfg, backbone=b2)
        load_teacher_state(m2, blob)
        # copy dummy backbone weights too so tokens match
        m2.backbone.load_state_dict(m.backbone.state_dict())
        with torch.no_grad():
            c = m2(x)["Hhat_1s"]
        self.assertTrue(torch.allclose(a, c, atol=1e-5))

    def test_optimizer_resume_state(self) -> None:
        cfg = Config.load()
        b = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=b)
        groups = build_param_groups(m.adapter_parameters(), [], 1e-3, 1e-4, 1e-4)
        opt = torch.optim.AdamW(groups)
        x = torch.rand(1, 16, 3, 64, 64)
        loss = m(x)["Hhat_now"].mean()
        loss.backward()
        opt.step()
        state = opt.state_dict()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "opt.pt"
            save_checkpoint({"optimizer": state, **extract_teacher_state(m)}, path)
            blob = load_checkpoint(path)
        opt2 = torch.optim.AdamW(build_param_groups(m.adapter_parameters(), [], 1e-3, 1e-4, 1e-4))
        opt2.load_state_dict(blob["optimizer"])
        self.assertEqual(opt2.state_dict()["state"].keys(), state["state"].keys())

    def test_lora_roundtrip(self) -> None:
        cfg = Config.load()
        b = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16, num_layers=8)
        m = CollisionTeacher3(cfg, backbone=b)
        m.enable_lora()
        with torch.no_grad():
            for p in m.lora_parameters():
                p.add_(0.01)
        blob = extract_teacher_state(m)
        b2 = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16, num_layers=8)
        m2 = CollisionTeacher3(cfg, backbone=b2)
        m2.enable_lora()
        load_teacher_state(m2, blob)
        for p1, p2 in zip(m.lora_parameters(), m2.lora_parameters()):
            self.assertTrue(torch.allclose(p1, p2))


if __name__ == "__main__":
    unittest.main()
