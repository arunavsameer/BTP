"""V3 extras: CoordConv + Δ1/Δ3 + time embed. Does not import-train V2 runs."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.v2.model import CollisionStudentV2, n_params
from student.v3.model import STEM_C, CollisionStudentV3, causal_delta, coord_xyr


class DummyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Conv2d(128, 1, kernel_size=1)

    def forward_fine_logits(self, z):
        return self.net(z)

    def forward_checkpoints(self, z):
        logits_fine = self.forward_fine_logits(z)
        pooled = torch.nn.functional.adaptive_max_pool2d(logits_fine, 5)
        p = torch.sigmoid(pooled).squeeze(1)
        return {"Z_16": logits_fine, "P_5": p, "Hhat_1s": p, "Hhat_1s_logits": pooled.squeeze(1)}


class StudentV3Tests(unittest.TestCase):
    def test_trainable_param_count_near_v2_plus_extras(self):
        v2 = CollisionStudentV2(DummyDecoder())
        v3 = CollisionStudentV3(DummyDecoder())
        n2 = n_params(v2, trainable=True)
        n3 = n_params(v3, trainable=True)
        extra = n3 - n2
        self.assertGreaterEqual(n3, 390_000)
        self.assertLessEqual(n3, 400_000)
        # Δ3 96→128: 12288, coord 3→128: 384, time embed: 1024
        self.assertGreaterEqual(extra, 13_000)
        self.assertLessEqual(extra, 14_000)
        self.assertEqual(int(v3.backbone.time_embed.numel()), 1024)
        self.assertEqual(int(v3.backbone.vel_fuse.conv.in_channels), STEM_C * 3 + 3)
        self.assertFalse(any(p.requires_grad for p in v3.decoder.parameters()))

    def test_causality_no_future_leak_into_t0_t6(self):
        model = CollisionStudentV3(DummyDecoder()).eval()
        clip_a = torch.rand(1, 16, 3, 256, 256)
        clip_b = clip_a.clone()
        clip_b[:, 14:] = torch.rand_like(clip_b[:, 14:])
        with torch.no_grad():
            xa = model.normalize(clip_a).permute(0, 2, 1, 3, 4)
            xb = model.normalize(clip_b).permute(0, 2, 1, 3, 4)
            ma = model.backbone(xa)["F_motion"]
            mb = model.backbone(xb)["F_motion"]
        err = (ma[:, :, :7] - mb[:, :, :7]).abs().max().item()
        self.assertLess(err, 1e-6, f"future leak {err}")

    def test_coord_and_delta3_are_causal_priors(self):
        c = coord_xyr(16)
        self.assertEqual(tuple(c.shape), (1, 3, 1, 16, 16))
        self.assertTrue(torch.all(c[:, 0].abs() <= 1.0 + 1e-5))
        self.assertTrue(torch.all(c[:, 1].abs() <= 1.0 + 1e-5))
        self.assertTrue(torch.all(c[:, 2] >= 0))
        f = torch.randn(2, 96, 8, 16, 16)
        d3 = causal_delta(f, 3)
        self.assertEqual(tuple(d3.shape), tuple(f.shape))
        self.assertTrue(torch.equal(d3[:, :, :3], torch.zeros_like(d3[:, :, :3])))
        self.assertTrue(torch.allclose(d3[:, :, 3], f[:, :, 3] - f[:, :, 0]))


if __name__ == "__main__":
    unittest.main()
