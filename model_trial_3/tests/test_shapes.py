from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.models.heatmap_decoder import HeatmapDecoder, pool_heatmap
from collision_jepa3.models.spatial_adapter import SpatialTemporalAdapter
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone


class ShapeTests(unittest.TestCase):
    def test_adapter_z_shape(self) -> None:
        ad = SpatialTemporalAdapter(hidden_size=32, temporal_groups=8, spatial=4, z_channels=32, feature_grid=5, projection_channels=32)
        tokens = torch.randn(2, 8 * 4 * 4, 32)
        z = ad(tokens)
        self.assertEqual(tuple(z.shape), (2, 32, 5, 5))
        z2, fine = ad.features(tokens)
        self.assertTrue(torch.equal(z, z2))
        self.assertEqual(tuple(fine.shape), (2, 32, 4, 4))
        self.assertEqual(ad.temporal_pool, "softmax")
        self.assertEqual(len(ad.temporal), 4)

    def test_temporal_softmax_sums_to_one(self) -> None:
        ad = SpatialTemporalAdapter(
            hidden_size=32, temporal_groups=8, spatial=4, z_channels=32, feature_grid=5, projection_channels=32
        )
        tokens = torch.randn(2, 8 * 4 * 4, 32)
        grid = ad.tokens_to_grid(tokens)
        b, t_prime, c, s, _ = grid.shape
        x = grid.reshape(b * t_prime, c, s, s)
        x = ad.spatial_proj(x)
        x = x.reshape(b, t_prime, x.shape[1], s, s).permute(0, 2, 1, 3, 4).contiguous()
        x = ad.temporal(x)
        attn = ad.time_pool.attention(x)
        self.assertEqual(tuple(attn.shape), (2, 1, 8, 4, 4))
        self.assertTrue(torch.allclose(attn.sum(dim=2), torch.ones(2, 1, 4, 4), atol=1e-5))

    def test_mean_last_pool_still_available(self) -> None:
        ad = SpatialTemporalAdapter(
            hidden_size=32,
            temporal_groups=8,
            spatial=4,
            z_channels=32,
            feature_grid=5,
            projection_channels=32,
            temporal_blocks=2,
            temporal_dilation=1,
            temporal_dilations=[1, 1],
            temporal_pool="mean_last",
        )
        z = ad(torch.randn(1, 8 * 4 * 4, 32))
        self.assertEqual(tuple(z.shape), (1, 32, 5, 5))
        self.assertIsNone(ad.time_pool)

    def test_decoder_dilated_independent_sigmoid(self) -> None:
        dec = HeatmapDecoder(z_channels=32)
        dilated = [m for m in dec.net.modules() if isinstance(m, torch.nn.Conv2d) and m.dilation == (2, 2)]
        self.assertEqual(len(dilated), 1)
        h = dec(torch.randn(3, 32, 5, 5)).detach()
        self.assertEqual(tuple(h.shape), (3, 5, 5))
        self.assertGreaterEqual(float(h.min()), 0.0)
        self.assertLessEqual(float(h.max()), 1.0)
        # Per-cell sigmoid, not a spatial softmax over the 25 cells.
        self.assertFalse(torch.allclose(h.reshape(3, -1).sum(-1), torch.ones(3), atol=1e-3))

    def test_max_pool_keeps_a_small_peak(self) -> None:
        h = torch.zeros(1, 16, 16)
        h[0, 0, 0] = 1.0
        mx = pool_heatmap(h, 5, "max")
        av = pool_heatmap(h, 5, "avg")
        self.assertEqual(tuple(mx.shape), (1, 5, 5))
        self.assertAlmostEqual(float(mx.max()), 1.0, places=5)
        self.assertLess(float(av.max()), 0.2)

    def test_fine_decoder_pools_to_5(self) -> None:
        dec = HeatmapDecoder(z_channels=32, out_grid=5, pool="max")
        h = dec(torch.randn(2, 32, 16, 16)).detach()
        self.assertEqual(tuple(h.shape), (2, 5, 5))
        self.assertGreaterEqual(float(h.min()), 0.0)
        self.assertLessEqual(float(h.max()), 1.0)

    def test_attn_pool_to_5(self) -> None:
        dec = HeatmapDecoder(z_channels=32, out_grid=5, pool="attn")
        h = dec(torch.randn(1, 32, 16, 16)).detach()
        self.assertEqual(tuple(h.shape), (1, 5, 5))

    def test_teacher_outputs(self) -> None:
        cfg = Config.load()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        clip = torch.rand(2, 16, 3, 64, 64)
        out = m(clip)
        self.assertEqual(tuple(out["Z_t"].shape), (2, 32, 5, 5))
        self.assertEqual(m.decode_from, "fine")
        self.assertEqual(m.decoder.net[0].in_channels, 128)
        self.assertEqual(tuple(out["Hhat_now_logits"].shape), (2, 5, 5))
        for k in ("Hhat_now", "Hhat_1s", "Hhat_1_5s", "Hhat_2s"):
            self.assertEqual(tuple(out[k].shape), (2, 5, 5))
            self.assertGreaterEqual(float(out[k].detach().min()), 0.0)
            self.assertLessEqual(float(out[k].detach().max()), 1.0)
        for k in ("Delta_1s", "Delta_1_5s", "Delta_2s"):
            self.assertEqual(tuple(out[k].shape), (2, 5, 5))
            self.assertGreaterEqual(float(out[k].detach().min()), -1.0)
            self.assertLessEqual(float(out[k].detach().max()), 1.0)

    def test_direct_h1s_outputs(self) -> None:
        cfg = Config.load()
        cfg.set("heads.mode", "direct_h1s")
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        out = m(torch.rand(2, 16, 3, 64, 64))
        self.assertEqual(set(out), {"Z_t", "Hhat_1s", "Hhat_1s_logits"})
        self.assertEqual(tuple(out["Z_t"].shape), (2, 32, 5, 5))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (2, 5, 5))

    def test_decode_from_z_uses_32ch_decoder(self) -> None:
        cfg = Config.load()
        cfg.set("adapter.decode_from", "z")
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        m = CollisionTeacher3(cfg, backbone=backbone)
        self.assertEqual(m.decoder.net[0].in_channels, 32)
        out = m(torch.rand(1, 16, 3, 64, 64))
        self.assertEqual(tuple(out["Hhat_now"].shape), (1, 5, 5))
        self.assertEqual(tuple(out["Z_t"].shape), (1, 32, 5, 5))

    def test_wrong_token_count_fails(self) -> None:
        ad = SpatialTemporalAdapter(hidden_size=32, temporal_groups=8, spatial=4)
        tokens = torch.randn(1, 100, 32)
        with self.assertRaises(RuntimeError):
            ad(tokens)


if __name__ == "__main__":
    unittest.main()
