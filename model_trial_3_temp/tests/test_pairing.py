from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.manifest import ClipRecord
from collision_jepa3.data.paired_sampler import background_partner_id, outcome_partner_ids
from collision_jepa3.data.splits import split_groups


def _rec(sample, outcome, bg, valid=True) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{sample}/{outcome}/{bg}",
        sample_id=sample,
        scenario_group_id=sample,
        scenario_type=sample.split("_", 2)[-1],
        outcome_type=outcome,
        background_id=bg,
        video_path="",
        episode_json_path="",
        heatmap_json_path="",
        frame_count=450,
        fps=30,
        width=1920,
        height=1080,
        valid=valid,
        outcome_group_id=f"{sample}|{bg}",
        background_pair_id=f"{sample}|{outcome}",
    )


class PairingTests(unittest.TestCase):
    def test_background_partners_match_ids(self) -> None:
        recs = [
            _rec("sample_0006_cube_from_left", "collision", "bg_00"),
            _rec("sample_0006_cube_from_left", "collision", "bg_01"),
            _rec("sample_0006_cube_from_left", "miss_left", "bg_00"),
        ]
        partner = background_partner_id(recs[0], recs)
        self.assertEqual(partner, recs[1].clip_id)
        self.assertNotEqual(partner, recs[2].clip_id)

    def test_outcome_partners(self) -> None:
        recs = [
            _rec("sample_0006_cube_from_left", "collision", "bg_00"),
            _rec("sample_0006_cube_from_left", "miss_left", "bg_00"),
            _rec("sample_0006_cube_from_left", "collision", "bg_01"),
        ]
        mates = outcome_partner_ids(recs[0], recs)
        self.assertEqual(mates, [recs[1].clip_id])

    def test_split_keeps_groups_together(self) -> None:
        recs = []
        for i, scen in enumerate(["jaywalker", "cube", "car"] * 3):
            sample = f"sample_{i:04d}_{scen}"
            recs += [
                _rec(sample, "collision", "bg_00"),
                _rec(sample, "miss_left", "bg_00"),
                _rec(sample, "collision", "bg_01"),
            ]
        splits = split_groups(recs, seed=1337)
        group_split = {}
        for split, groups in splits.items():
            for g in groups:
                group_split[g] = split
        for rec in recs:
            self.assertEqual(group_split[rec.scenario_group_id], group_split[rec.scenario_group_id])
        # no group in two splits
        all_g = [g for gs in splits.values() for g in gs]
        self.assertEqual(len(all_g), len(set(all_g)))

    def test_small_unique_scenarios_still_get_val(self) -> None:
        recs = []
        for i, scen in enumerate(["a", "b", "c", "d", "e", "f"]):
            sample = f"sample_{i:04d}_{scen}"
            recs.append(_rec(sample, "collision", "bg_00"))
        splits = split_groups(recs, seed=1337)
        self.assertGreaterEqual(len(splits["train"]), 1)
        self.assertGreaterEqual(len(splits["val"]), 1)
        self.assertGreaterEqual(len(splits["test"]), 1)


if __name__ == "__main__":
    unittest.main()
