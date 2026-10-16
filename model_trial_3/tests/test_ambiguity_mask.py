from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.manifest import ClipRecord
from collision_jepa3.data.heatmap_annotations import targets_for_anchor
from collision_jepa3.data.pairing import (
    apply_mean_future_targets,
    compute_ambiguity_table,
    outcome_clip_ids_by_group,
)
from collision_jepa3.data.sampling import FRAME_OFFSETS, HORIZONS_FRAMES


def _rec(sample, outcome, bg) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{sample}/{outcome}/{bg}",
        sample_id=sample,
        scenario_group_id=sample,
        scenario_type="cube",
        outcome_type=outcome,
        background_id=bg,
        video_path="",
        episode_json_path="",
        heatmap_json_path="",
        frame_count=450,
        fps=30,
        width=64,
        height=64,
        valid=True,
        outcome_group_id=f"{sample}|{bg}",
        background_pair_id=f"{sample}|{outcome}",
        fork_frame=365,
    )


class AmbiguityTests(unittest.TestCase):
    def test_masks_only_affected_horizon(self) -> None:
        col = _rec("sample_0000", "collision", "bg_00")
        miss = _rec("sample_0000", "miss_left", "bg_00")
        hc = np.zeros((450, 5, 5), dtype=np.float32)
        hm = np.zeros((450, 5, 5), dtype=np.float32)
        # After fork, 2s target diverges for anchor 330: 330+60=390 > 365.
        # 1s target 360 < 365 still identical.
        hm[390:, :, :] = 1.0
        table = compute_ambiguity_table(
            {col.clip_id: hc, miss.clip_id: hm},
            [col, miss],
            fork_frame=365,
            n_frames=450,
            stride=15,
            offsets=FRAME_OFFSETS,
            horizons=HORIZONS_FRAMES,
            horizon_names=["1s", "1_5s", "2s"],
            heatmap_tol=0.02,
            mask_ambiguous=True,
        )
        # t=330 is a valid stride-15 anchor (90,105,...,330,...)
        flags = table[(col.clip_id, 330)]
        self.assertFalse(flags["1s"])  # 360 still pre-fork identical zeros
        self.assertTrue(flags["2s"])  # 390 post-fork disagrees
        # After the fork, history includes post-fork RGB so do not mask.
        flags_post = table[(col.clip_id, 375)]
        self.assertFalse(flags_post["2s"])

    def test_identical_futures_not_masked(self) -> None:
        col = _rec("sample_0000", "collision", "bg_00")
        miss = _rec("sample_0000", "miss_left", "bg_00")
        h = np.zeros((450, 5, 5), dtype=np.float32)
        table = compute_ambiguity_table(
            {col.clip_id: h, miss.clip_id: h.copy()},
            [col, miss],
            fork_frame=365,
            n_frames=450,
            stride=15,
            offsets=FRAME_OFFSETS,
            horizons=HORIZONS_FRAMES,
            horizon_names=["1s", "1_5s", "2s"],
            heatmap_tol=0.02,
            mask_ambiguous=True,
        )
        flags = table[(col.clip_id, 300)]
        self.assertFalse(any(flags.values()))

    def test_mean_target_rewrites_miss_future(self) -> None:
        col = _rec("sample_0000", "collision", "bg_00")
        miss = _rec("sample_0000", "miss_left", "bg_00")
        hc = np.zeros((450, 5, 5), dtype=np.float32)
        hm = np.zeros((450, 5, 5), dtype=np.float32)
        hc[390, 2, 2] = 0.96
        hm[390:, :, :] = 1.0
        heatmaps = {col.clip_id: hc, miss.clip_id: hm}
        table = compute_ambiguity_table(
            heatmaps,
            [col, miss],
            fork_frame=365,
            n_frames=450,
            stride=15,
            offsets=FRAME_OFFSETS,
            horizons=HORIZONS_FRAMES,
            horizon_names=["1s", "1_5s", "2s"],
            heatmap_tol=0.02,
            mask_ambiguous=True,
        )
        tg = targets_for_anchor(hm, 330, HORIZONS_FRAMES)
        out, replaced = apply_mean_future_targets(
            tg,
            rec=miss,
            anchor=330,
            amb=table[(miss.clip_id, 330)],
            clip_heatmaps=heatmaps,
            outcome_ids=outcome_clip_ids_by_group([col, miss]),
            horizons=HORIZONS_FRAMES,
            horizon_names=["1s", "1_5s", "2s"],
        )
        self.assertIn("2s", replaced)
        self.assertAlmostEqual(float(out["h_2s"][2, 2]), 0.98)


if __name__ == "__main__":
    unittest.main()
