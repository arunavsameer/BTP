"""Call pairing.py functions on synthetic and real clips. Does not change pairing.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.heatmap_annotations import load_heatmaps, targets_for_anchor
from collision_jepa3.data.manifest import ClipRecord
from collision_jepa3.data.pairing import (
    apply_mean_future_targets,
    background_mates,
    background_pair_valid,
    compute_ambiguity_table,
    heatmaps_disagree,
    outcome_clip_ids_by_group,
    outcome_mates,
)
from collision_jepa3.data.sampling import FRAME_OFFSETS, HORIZONS_FRAMES

WORKSPACE = ROOT.parent
REAL_DATASET = WORKSPACE / "blender_sim" / "datasets" / "groups15_n400_s67_n400"
SAMPLE = "sample_0000_jaywalker"
HORIZON_NAMES = ["1s", "1_5s", "2s"]


def _rec(sample: str, outcome: str, bg: str, *, valid: bool = True, fork: int = 365) -> ClipRecord:
    return ClipRecord(
        clip_id=f"{sample}/{outcome}/{bg}",
        sample_id=sample,
        scenario_group_id=sample,
        scenario_type=sample.split("_", 2)[-1] if sample.startswith("sample_") else sample,
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
        fork_frame=fork,
    )


def _zeros() -> np.ndarray:
    return np.zeros((450, 5, 5), dtype=np.float32)


def _ambiguity(
    heatmaps: dict[str, np.ndarray],
    records: list[ClipRecord],
    *,
    mask: bool = True,
    tol: float = 0.02,
):
    return compute_ambiguity_table(
        heatmaps,
        records,
        fork_frame=365,
        n_frames=450,
        stride=15,
        offsets=FRAME_OFFSETS,
        horizons=HORIZONS_FRAMES,
        horizon_names=HORIZON_NAMES,
        heatmap_tol=tol,
        mask_ambiguous=mask,
    )


class HeatmapsDisagreeTests(unittest.TestCase):
    def test_identical_maps_agree(self) -> None:
        a = np.zeros((5, 5), dtype=np.float32)
        self.assertFalse(heatmaps_disagree(a, a.copy(), 0.02))

    def test_small_difference_within_tol_agrees(self) -> None:
        a = np.zeros((5, 5), dtype=np.float32)
        b = a.copy()
        b[2, 2] = 0.01
        self.assertFalse(heatmaps_disagree(a, b, 0.02))

    def test_difference_above_tol_disagrees(self) -> None:
        a = np.zeros((5, 5), dtype=np.float32)
        b = a.copy()
        b[2, 2] = 0.03
        self.assertTrue(heatmaps_disagree(a, b, 0.02))


class MateGroupingTests(unittest.TestCase):
    def test_outcome_mates_share_sample_and_background(self) -> None:
        recs = [
            _rec(SAMPLE, "collision", "bg_00"),
            _rec(SAMPLE, "miss_left", "bg_00"),
            _rec(SAMPLE, "collision", "bg_01"),
            _rec(SAMPLE, "miss_left", "bg_01"),
            _rec(SAMPLE, "collision", "bg_00", valid=False),
        ]
        groups = outcome_mates(recs)
        ids = {r.clip_id for r in groups[f"{SAMPLE}|bg_00"]}
        self.assertEqual(
            ids,
            {f"{SAMPLE}/collision/bg_00", f"{SAMPLE}/miss_left/bg_00"},
        )
        self.assertNotIn(f"{SAMPLE}/collision/bg_01", ids)

    def test_background_mates_share_sample_and_outcome(self) -> None:
        recs = [
            _rec(SAMPLE, "collision", "bg_00"),
            _rec(SAMPLE, "collision", "bg_01"),
            _rec(SAMPLE, "miss_left", "bg_00"),
        ]
        groups = background_mates(recs)
        ids = {r.clip_id for r in groups[f"{SAMPLE}|collision"]}
        self.assertEqual(
            ids,
            {f"{SAMPLE}/collision/bg_00", f"{SAMPLE}/collision/bg_01"},
        )


class AmbiguityTableTests(unittest.TestCase):
    def test_mask_off_never_flags(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        hc, hm = _zeros(), _zeros()
        hm[390:] = 1.0
        table = _ambiguity({col.clip_id: hc, miss.clip_id: hm}, [col, miss], mask=False)
        self.assertFalse(any(table[(col.clip_id, 330)].values()))

    def test_pre_fork_masks_only_horizons_that_disagree(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        hc, hm = _zeros(), _zeros()
        # Anchor 330: 1s=360 still pre-fork; 1.5s=375 and 2s=390 post-fork.
        hm[375:] = 1.0
        table = _ambiguity({col.clip_id: hc, miss.clip_id: hm}, [col, miss])
        flags = table[(col.clip_id, 330)]
        miss_flags = table[(miss.clip_id, 330)]
        self.assertFalse(flags["1s"])
        self.assertTrue(flags["1_5s"])
        self.assertTrue(flags["2s"])
        self.assertEqual(flags, miss_flags)

    def test_post_fork_anchor_is_not_masked(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        hc, hm = _zeros(), _zeros()
        hm[390:] = 1.0
        table = _ambiguity({col.clip_id: hc, miss.clip_id: hm}, [col, miss])
        self.assertFalse(any(table[(col.clip_id, 375)].values()))

    def test_identical_futures_are_not_masked(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        h = _zeros()
        table = _ambiguity({col.clip_id: h, miss.clip_id: h.copy()}, [col, miss])
        self.assertFalse(any(table[(col.clip_id, 300)].values()))

    def test_solo_clip_is_not_masked(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        h = _zeros()
        h[390:] = 1.0
        table = _ambiguity({col.clip_id: h}, [col])
        self.assertFalse(any(table[(col.clip_id, 330)].values()))

    def test_invalid_clip_is_skipped(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00", valid=False)
        miss = _rec(SAMPLE, "miss_left", "bg_00", valid=False)
        table = _ambiguity({}, [col, miss])
        self.assertEqual(table, {})

    def test_both_misses_can_be_masked(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        left = _rec(SAMPLE, "miss_left", "bg_00")
        right = _rec(SAMPLE, "miss_right", "bg_00")
        hc, hl, hr = _zeros(), _zeros(), _zeros()
        hl[390:] = 1.0
        hr[390:] = 0.8
        table = _ambiguity(
            {col.clip_id: hc, left.clip_id: hl, right.clip_id: hr},
            [col, left, right],
        )
        self.assertTrue(table[(col.clip_id, 330)]["2s"])
        self.assertTrue(table[(left.clip_id, 330)]["2s"])
        self.assertTrue(table[(right.clip_id, 330)]["2s"])


class CollisionFutureTargetTests(unittest.TestCase):
    def test_miss_2s_is_replaced_with_mean_heatmap(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        hc, hm = _zeros(), _zeros()
        hc[390, 2, 2] = 0.96
        hm[390] = 0.0
        heatmaps = {col.clip_id: hc, miss.clip_id: hm}
        table = _ambiguity(heatmaps, [col, miss])
        amb = table[(miss.clip_id, 330)]
        self.assertTrue(amb["2s"])
        self.assertFalse(amb["1s"])
        tg = targets_for_anchor(hm, 330, HORIZONS_FRAMES)
        out, replaced = apply_mean_future_targets(
            tg,
            rec=miss,
            anchor=330,
            amb=amb,
            clip_heatmaps=heatmaps,
            outcome_ids=outcome_clip_ids_by_group([col, miss]),
            horizons=HORIZONS_FRAMES,
            horizon_names=HORIZON_NAMES,
        )
        self.assertEqual(replaced, {"2s"})
        self.assertAlmostEqual(float(out["h_2s"][2, 2]), 0.48)
        self.assertFalse(np.allclose(out["h_2s"], tg["h_2s"]))
        np.testing.assert_array_equal(out["h_1s"], tg["h_1s"])

    def test_collision_clip_also_gets_mean_heatmap(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        hc, hm = _zeros(), _zeros()
        hc[390, 2, 2] = 0.96
        heatmaps = {col.clip_id: hc, miss.clip_id: hm}
        table = _ambiguity(heatmaps, [col, miss])
        tg = targets_for_anchor(hc, 330, HORIZONS_FRAMES)
        out, replaced = apply_mean_future_targets(
            tg,
            rec=col,
            anchor=330,
            amb=table[(col.clip_id, 330)],
            clip_heatmaps=heatmaps,
            outcome_ids=outcome_clip_ids_by_group([col, miss]),
            horizons=HORIZONS_FRAMES,
            horizon_names=HORIZON_NAMES,
        )
        self.assertIn("2s", replaced)
        self.assertAlmostEqual(float(out["h_2s"][2, 2]), 0.48)

    def test_post_fork_is_not_replaced(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        hc, hm = _zeros(), _zeros()
        hc[390, 2, 2] = 0.96
        heatmaps = {col.clip_id: hc, miss.clip_id: hm}
        table = _ambiguity(heatmaps, [col, miss])
        tg = targets_for_anchor(hm, 375, HORIZONS_FRAMES)
        out, replaced = apply_mean_future_targets(
            tg,
            rec=miss,
            anchor=375,
            amb=table[(miss.clip_id, 375)],
            clip_heatmaps=heatmaps,
            outcome_ids=outcome_clip_ids_by_group([col, miss]),
            horizons=HORIZONS_FRAMES,
            horizon_names=HORIZON_NAMES,
        )
        self.assertEqual(replaced, set())
        np.testing.assert_array_equal(out["h_2s"], tg["h_2s"])


class BackgroundPairValidTests(unittest.TestCase):
    def test_same_outcome_different_bg_with_matching_heatmaps(self) -> None:
        a = _rec(SAMPLE, "collision", "bg_00")
        b = _rec(SAMPLE, "collision", "bg_01")
        h = _zeros()
        self.assertTrue(background_pair_valid(a, b, h, h.copy(), 300, HORIZONS_FRAMES, 0.001))

    def test_rejects_different_outcome(self) -> None:
        a = _rec(SAMPLE, "collision", "bg_00")
        b = _rec(SAMPLE, "miss_left", "bg_00")
        h = _zeros()
        self.assertFalse(background_pair_valid(a, b, h, h.copy(), 300, HORIZONS_FRAMES, 0.001))

    def test_rejects_same_clip(self) -> None:
        a = _rec(SAMPLE, "collision", "bg_00")
        h = _zeros()
        self.assertFalse(background_pair_valid(a, a, h, h.copy(), 300, HORIZONS_FRAMES, 0.001))

    def test_rejects_disagreeing_future_heatmaps(self) -> None:
        a = _rec(SAMPLE, "collision", "bg_00")
        b = _rec(SAMPLE, "collision", "bg_01")
        ha, hb = _zeros(), _zeros()
        hb[330] = 1.0  # now-target at anchor 300 is 300; 1s is 330
        self.assertFalse(background_pair_valid(a, b, ha, hb, 300, HORIZONS_FRAMES, 0.001))

    def test_late_anchor_ignores_missing_2s(self) -> None:
        a = _rec(SAMPLE, "collision", "bg_00")
        b = _rec(SAMPLE, "collision", "bg_01")
        h = _zeros()
        # t=419: now and 1s exist; 1.5s/2s are past frame 449.
        self.assertTrue(background_pair_valid(a, b, h, h.copy(), 419, HORIZONS_FRAMES, 0.001))


class RealClipPairingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.col_dir = REAL_DATASET / SAMPLE / "collision" / "bg_00"
        self.miss_dir = REAL_DATASET / SAMPLE / "miss_left" / "bg_00"
        self.col_bg1 = REAL_DATASET / SAMPLE / "collision" / "bg_01"
        hm_name = Path("spatial_annotations") / "spatial_annotations.json"
        needed = [
            self.col_dir / hm_name,
            self.miss_dir / hm_name,
            self.col_bg1 / hm_name,
        ]
        if not all(p.exists() for p in needed):
            self.skipTest("real paired heatmaps missing")
        self.h_col = load_heatmaps(self.col_dir / hm_name, expected_frames=450, grid=5)
        self.h_miss = load_heatmaps(self.miss_dir / hm_name, expected_frames=450, grid=5)
        self.h_col_bg1 = load_heatmaps(self.col_bg1 / hm_name, expected_frames=450, grid=5)

    def test_real_collision_and_miss_are_ambiguous_at_2s(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        table = _ambiguity(
            {col.clip_id: self.h_col, miss.clip_id: self.h_miss},
            [col, miss],
        )
        flags = table[(col.clip_id, 330)]
        self.assertTrue(flags["2s"], "2s target 390 is after fork 365, heatmaps should differ")
        self.assertFalse(any(table[(col.clip_id, 375)].values()))

    def test_real_miss_2s_uses_mean_heatmap(self) -> None:
        col = _rec(SAMPLE, "collision", "bg_00")
        miss = _rec(SAMPLE, "miss_left", "bg_00")
        heatmaps = {col.clip_id: self.h_col, miss.clip_id: self.h_miss}
        table = _ambiguity(heatmaps, [col, miss])
        tg = targets_for_anchor(self.h_miss, 330, HORIZONS_FRAMES)
        out, replaced = apply_mean_future_targets(
            tg,
            rec=miss,
            anchor=330,
            amb=table[(miss.clip_id, 330)],
            clip_heatmaps=heatmaps,
            outcome_ids=outcome_clip_ids_by_group([col, miss]),
            horizons=HORIZONS_FRAMES,
            horizon_names=HORIZON_NAMES,
        )
        self.assertIn("2s", replaced)
        expected = 0.5 * (self.h_col[390] + self.h_miss[390])
        np.testing.assert_allclose(out["h_2s"], expected, atol=1e-5)

    def test_real_background_pair_heatmaps_match(self) -> None:
        a = _rec(SAMPLE, "collision", "bg_00")
        b = _rec(SAMPLE, "collision", "bg_01")
        self.assertTrue(
            background_pair_valid(a, b, self.h_col, self.h_col_bg1, 300, HORIZONS_FRAMES, 0.001)
        )


if __name__ == "__main__":
    unittest.main()
