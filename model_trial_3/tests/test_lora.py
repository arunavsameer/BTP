from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.models.lora import LoRALinear, apply_lora_to_last_blocks, lora_parameters
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone


class LoRATests(unittest.TestCase):
    def test_lora_linear(self) -> None:
        lin = nn.Linear(6, 4)
        lora = LoRALinear(lin, rank=2, alpha=4.0)
        y = lora(torch.randn(3, 6))
        self.assertEqual(tuple(y.shape), (3, 4))
        self.assertFalse(lin.weight.requires_grad)
        self.assertTrue(lora.lora_A.requires_grad)

    def test_only_lora_and_heads_trainable(self) -> None:
        cfg = Config.load()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, num_frames=16, num_layers=8)
        m = CollisionTeacher3(cfg, backbone=backbone)
        n = m.enable_lora()
        self.assertGreater(n, 0)
        lora_ids = {id(p) for p in m.lora_parameters()}
        adapter_ids = {id(p) for p in m.adapter_parameters()}
        for p in m.parameters():
            if id(p) in lora_ids or id(p) in adapter_ids:
                self.assertTrue(p.requires_grad)
            else:
                self.assertFalse(p.requires_grad)

    def test_missing_blocks_raises(self) -> None:
        class Empty(nn.Module):
            def __init__(self):
                super().__init__()
                self.foo = nn.Linear(4, 4)

        with self.assertRaises(RuntimeError):
            apply_lora_to_last_blocks(Empty())


if __name__ == "__main__":
    unittest.main()
