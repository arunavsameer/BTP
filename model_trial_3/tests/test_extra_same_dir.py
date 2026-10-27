from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.extra_same_dir import (
    apply_extra_same_dir_splits,
    extra_same_dir_groups,
    sample_index,
)
from collision_jepa3.data.manifest import ClipRecord


def _rec(gid: str, valid: bool = True) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{gid}/miss_left/bg_00",
        sample_id=gid,
        scenario_group_id=gid,
        scenario_type="same_dir_hold",
        outcome_type="miss_left",
        background_id="bg_00",
        video_path="",
        episode_json_path="",
        heatmap_json_path="",
        frame_count=450,
        fps=30.0,
        width=1920,
        height=1080,
        valid=valid,
    )


class ExtraSameDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config.load()

    def test_sample_index(self) -> None:
        self.assertEqual(sample_index("sample_0207_same_dir_closing"), 207)
        self.assertEqual(sample_index("sample_0229_same_dir_pulling"), 229)
        self.assertIsNone(sample_index("sample_jaywalker"))

    def test_hold_only_range_drops_closing(self) -> None:
        recs = [
            _rec("sample_0207_same_dir_closing"),
            _rec("sample_0210_same_dir_hold"),
            _rec("sample_0219_same_dir_hold"),
            _rec("sample_0220_same_dir_pulling"),
        ]
        self.cfg.set("data.extra_same_dir.sample_min", 210)
        self.cfg.set("data.extra_same_dir.sample_max", 219)
        g = extra_same_dir_groups(recs, self.cfg)
        self.assertEqual(g, {"sample_0210_same_dir_hold", "sample_0219_same_dir_hold"})

    def test_range_excludes_200_206(self) -> None:
        recs = [
            _rec("sample_0200_same_dir_closing"),
            _rec("sample_0206_same_dir_closing"),
            _rec("sample_0207_same_dir_closing"),
            _rec("sample_0229_same_dir_pulling"),
            _rec("sample_0003_jaywalker_turn_toward"),
        ]
        g = extra_same_dir_groups(recs, self.cfg)
        self.assertEqual(g, {"sample_0207_same_dir_closing", "sample_0229_same_dir_pulling"})

    def test_enabled_adds_train_only(self) -> None:
        recs = [_rec("sample_0210_same_dir_hold"), _rec("sample_0003_jaywalker_turn_toward")]
        splits = {
            "train": ["sample_0003_jaywalker_turn_toward"],
            "val": ["sample_0003_jaywalker_turn_toward"],
            "test": [],
        }
        self.cfg.set("data.extra_same_dir.enabled", True)
        out = apply_extra_same_dir_splits(splits, recs, self.cfg)
        self.assertIn("sample_0210_same_dir_hold", out["train"])
        self.assertNotIn("sample_0210_same_dir_hold", out["val"])
        self.assertIn("sample_0003_jaywalker_turn_toward", out["val"])

    def test_disabled_strips_even_if_listed(self) -> None:
        recs = [_rec("sample_0210_same_dir_hold")]
        splits = {"train": ["sample_0210_same_dir_hold", "sample_0003_jaywalker_turn_toward"], "val": [], "test": []}
        self.cfg.set("data.extra_same_dir.enabled", False)
        out = apply_extra_same_dir_splits(splits, recs, self.cfg)
        self.assertNotIn("sample_0210_same_dir_hold", out["train"])
        self.assertIn("sample_0003_jaywalker_turn_toward", out["train"])


if __name__ == "__main__":
    unittest.main()
