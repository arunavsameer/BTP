from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.cache import TokenCache
from collision_jepa3.data.dataset import TeacherDataset, collate_teacher_batch
from collision_jepa3.data.paired_sampler import ClipContiguousSampler
from collision_jepa3.models.teacher import CollisionTeacher3
from collision_jepa3.models.vjepa_backbone import DummyVJEPA2Backbone
from collision_jepa3.training.engine import teacher_forward
from collision_jepa3.config import Config


class ClipContiguousSamplerTests(unittest.TestCase):
    def test_keeps_clip_blocks_and_shuffles_clip_order(self) -> None:
        index = [("a", t) for t in (90, 105, 120)] + [("b", t) for t in (90, 105)]
        s0 = list(ClipContiguousSampler(index, shuffle=False, seed=0))
        self.assertEqual(s0, [0, 1, 2, 3, 4])
        orders = {tuple(ClipContiguousSampler(index, shuffle=True, seed=seed)) for seed in range(8)}
        self.assertIn((0, 1, 2, 3, 4), orders)
        self.assertIn((3, 4, 0, 1, 2), orders)
        for seq in orders:
            a = list(seq).index(0)
            self.assertEqual(list(seq[a : a + 3]), [0, 1, 2])


class SubsampleTests(unittest.TestCase):
    def test_subsample_clips_is_seeded_and_keeps_all_anchors(self) -> None:
        ds = TeacherDataset.__new__(TeacherDataset)
        ds.index = [(f"c{i}", t) for i in range(8) for t in (90, 105, 120)]
        ds.subsample_clips(0.25, seed=1337)
        ids = ds.clip_ids()
        self.assertEqual(len(ids), 2)
        for cid in ids:
            anchors = [t for c, t in ds.index if c == cid]
            self.assertEqual(anchors, [90, 105, 120])
        ds2 = TeacherDataset.__new__(TeacherDataset)
        ds2.index = [(f"c{i}", t) for i in range(8) for t in (90, 105, 120)]
        ds2.subsample_clips(0.25, seed=1337)
        self.assertEqual(ds.clip_ids(), ds2.clip_ids())


class TokenCacheTests(unittest.TestCase):
    def test_roundtrip_and_teacher_forward(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cache = TokenCache(Path(td), "abcd")
            anchors = [90, 105]
            tokens = np.random.randn(2, 8, 16).astype(np.float32)
            cache.save_clip("sample/collision/bg_00", anchors, tokens)
            self.assertTrue(cache.has_clip("sample/collision/bg_00", anchors, 8, 16))
            arr, mapping = cache.load_clip("sample/collision/bg_00", mmap=True)
            self.assertEqual(mapping[105], 1)
            np.testing.assert_allclose(np.asarray(arr[0], dtype=np.float32), tokens[0], atol=2e-3)

        cfg = Config.load()
        backbone = DummyVJEPA2Backbone(hidden_size=64, crop_size=64, patch_size=16, tubelet_size=2, num_frames=16)
        model = CollisionTeacher3(cfg, backbone=backbone).eval()
        tok = torch.randn(2, 8 * 4 * 4, 64)
        batch = {"tokens": tok}
        out = teacher_forward(model, batch)
        self.assertEqual(tuple(out["Z_t"].shape), (2, 32, 5, 5))

    def test_collate_tokens_without_frames(self) -> None:
        samples = []
        for i in range(2):
            samples.append(
                {
                    "tokens": np.zeros((4, 8), dtype=np.float32),
                    "h_now": np.zeros((5, 5), dtype=np.float32),
                    "h_1s": np.zeros((5, 5), dtype=np.float32),
                    "h_1_5s": np.zeros((5, 5), dtype=np.float32),
                    "h_2s": np.zeros((5, 5), dtype=np.float32),
                    "anchor": np.int64(90 + i),
                    "bg_pair_valid": np.bool_(False),
                    "pre_fork": np.bool_(True),
                    "fork_frame": np.int64(365),
                    "amb_1s": np.bool_(False),
                    "amb_1_5s": np.bool_(False),
                    "amb_2s": np.bool_(False),
                    "clip_id": "a",
                    "sample_id": "a",
                    "scenario_group_id": "a",
                    "scenario_type": "a",
                    "outcome_type": "collision",
                    "background_id": "bg_00",
                    "outcome_group_id": "",
                    "background_pair_id": "",
                    "mate_clip_id": "",
                }
            )
        batch = collate_teacher_batch(samples)
        self.assertIn("tokens", batch)
        self.assertNotIn("frames", batch)
        self.assertEqual(tuple(batch["tokens"].shape), (2, 4, 8))


if __name__ == "__main__":
    unittest.main()
