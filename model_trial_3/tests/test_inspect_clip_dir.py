"""Call inspect_clip_dir on real and synthetic clips. Does not change manifest.py."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.manifest import inspect_clip_dir

WORKSPACE = ROOT.parent
REAL_DATASET = WORKSPACE / "blender_sim" / "datasets" / "groups15_n400_s67_n400"
REAL_CLIP = REAL_DATASET / "sample_0000_jaywalker" / "collision" / "bg_00"


def _write_heatmap(path: Path, n: int = 450, grid: int = 5, peak: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((grid, grid), dtype=np.float32)
    if peak:
        matrix[2, 2] = float(peak)
    frames = [{"frame_id": f"{i:06d}", "matrix": matrix.tolist()} for i in range(n)]
    path.write_text(json.dumps({"k": grid, "frames": frames}), encoding="utf-8")


def _write_video(path: Path, n_frames: int, size: tuple[int, int] = (64, 48), fps: float = 30.0) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def _write_episode(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_complete_clip(
    root: Path,
    *,
    sample: str = "sample_0001_cube_from_left",
    outcome: str = "collision",
    background: str = "bg_00",
    n_frames: int = 450,
    episode: dict | None = None,
    heatmap_peak: float = 0.25,
    dummy_video: bool = False,
) -> Path:
    clip = root / sample / outcome / background
    clip.mkdir(parents=True)
    video = clip / "preview.mp4"
    if dummy_video:
        video.write_bytes(b"not a real video")
    else:
        _write_video(video, n_frames)
    _write_episode(
        clip / "episode.json",
        episode
        or {
            "scenario": "cube_from_left",
            "diverge_frame": 365,
            "hit_s": 14.35,
            "video": "preview.mp4",
            "bg_lighting": "harsh_glare",
            "biome": "residential",
            "bg_style": "full",
            "ego": {"mode": "walk"},
            "label_histogram": {"SAFE_STATIC": 10},
        },
    )
    _write_heatmap(clip / "spatial_annotations" / "spatial_annotations.json", peak=heatmap_peak)
    (clip / "annotations").mkdir(exist_ok=True)
    (clip / "annotations" / "annotations.json").write_text("{}", encoding="utf-8")
    return clip


class InspectClipDirTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config.load()

    def test_real_dataset_clip_is_accepted(self) -> None:
        if not (REAL_CLIP / "preview.mp4").exists():
            self.skipTest(f"real clip missing: {REAL_CLIP}")
        rec = inspect_clip_dir(REAL_CLIP, REAL_DATASET, self.cfg)
        self.assertTrue(rec.valid, rec.invalid_reason)
        self.assertEqual(rec.clip_id, "sample_0000_jaywalker/collision/bg_00")
        self.assertEqual(rec.sample_id, "sample_0000_jaywalker")
        self.assertEqual(rec.scenario_group_id, "sample_0000_jaywalker")
        self.assertEqual(rec.scenario_type, "jaywalker")
        self.assertEqual(rec.outcome_type, "collision")
        self.assertEqual(rec.background_id, "bg_00")
        self.assertEqual(rec.lighting, "overcast")
        self.assertEqual(rec.biome, "park")
        self.assertEqual(rec.bg_style, "full")
        self.assertEqual(rec.diverge_frame, 365)
        self.assertEqual(rec.fork_frame, 365)
        self.assertAlmostEqual(float(rec.hit_s), 14.35)
        self.assertEqual(rec.outcome_group_id, "sample_0000_jaywalker|bg_00")
        self.assertEqual(rec.background_pair_id, "sample_0000_jaywalker|collision")
        self.assertGreaterEqual(rec.frame_count, 450)
        self.assertEqual(rec.width, 1920)
        self.assertEqual(rec.height, 1080)
        self.assertAlmostEqual(rec.fps, 30.0, places=0)
        self.assertTrue(Path(rec.video_path).is_file())
        self.assertTrue(Path(rec.episode_json_path).is_file())
        self.assertTrue(Path(rec.heatmap_json_path).is_file())
        self.assertIsNotNone(rec.overlay_path)
        self.assertIsNotNone(rec.annotations_path)
        self.assertEqual(rec.extra.get("scenario"), "jaywalker")
        self.assertEqual(rec.extra.get("ego_mode"), "walk")
        self.assertEqual(rec.extra.get("heatmap_shape"), [450, 5, 5])
        self.assertGreater(float(rec.extra.get("heatmap_peak", 0.0)), 0.0)
        self.assertIsNone(rec.invalid_reason)

    def test_incomplete_clip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "sample_0000_jaywalker" / "collision" / "bg_00"
            clip.mkdir(parents=True)
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("incomplete", rec.invalid_reason or "")
            self.assertIn("preview.mp4", rec.invalid_reason or "")
            self.assertIn("episode.json", rec.invalid_reason or "")
            self.assertEqual(rec.scenario_type, "jaywalker")
            self.assertEqual(rec.outcome_type, "collision")
            self.assertEqual(rec.background_id, "bg_00")
            self.assertEqual(rec.frame_count, 0)

    def test_missing_heatmap_only_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "sample_0002_cube" / "miss_left" / "bg_01"
            clip.mkdir(parents=True)
            (clip / "preview.mp4").write_bytes(b"x")
            _write_episode(clip / "episode.json", {"video": "preview.mp4"})
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("incomplete", rec.invalid_reason or "")
            self.assertIn("spatial_annotations/spatial_annotations.json", rec.invalid_reason or "")
            self.assertEqual(rec.outcome_type, "miss_left")
            self.assertEqual(rec.background_id, "bg_01")

    def test_complete_synthetic_clip_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = _make_complete_clip(root)
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertTrue(rec.valid, rec.invalid_reason)
            self.assertEqual(rec.scenario_type, "cube_from_left")
            self.assertEqual(rec.outcome_type, "collision")
            self.assertEqual(rec.background_id, "bg_00")
            self.assertEqual(rec.lighting, "harsh_glare")
            self.assertEqual(rec.biome, "residential")
            self.assertEqual(rec.bg_style, "full")
            self.assertEqual(rec.diverge_frame, 365)
            self.assertEqual(rec.fork_frame, 365)
            self.assertAlmostEqual(float(rec.hit_s), 14.35)
            self.assertEqual(rec.outcome_group_id, "sample_0001_cube_from_left|bg_00")
            self.assertEqual(rec.background_pair_id, "sample_0001_cube_from_left|collision")
            self.assertGreaterEqual(rec.frame_count, 450)
            self.assertEqual(rec.width, 64)
            self.assertEqual(rec.height, 48)
            self.assertEqual(rec.extra.get("scenario"), "cube_from_left")
            self.assertEqual(rec.extra.get("ego_mode"), "walk")
            self.assertEqual(rec.extra.get("heatmap_shape"), [450, 5, 5])
            self.assertAlmostEqual(float(rec.extra.get("heatmap_peak", 0.0)), 0.25, places=4)
            self.assertTrue(Path(rec.annotations_path).is_file())

    def test_lighting_falls_back_to_environment(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = _make_complete_clip(
                root,
                episode={
                    "scenario": "cube",
                    "video": "preview.mp4",
                    "environment": {"lighting": "dusk", "biome": "park"},
                    "ego": {"mode": "walk"},
                },
            )
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertTrue(rec.valid, rec.invalid_reason)
            self.assertEqual(rec.lighting, "dusk")
            self.assertEqual(rec.biome, "park")

    def test_overlay_video_leak_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = _make_complete_clip(
                root,
                dummy_video=True,
                episode={
                    "scenario": "cube",
                    "video": "spatial_overlay.mp4",
                    "spatial_overlay": "spatial_overlay.mp4",
                },
            )
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("episode.json", rec.invalid_reason or "")
            self.assertIn("overlay", (rec.invalid_reason or "").lower())

    def test_corrupt_episode_json_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "sample_0003_cube" / "collision" / "bg_00"
            clip.mkdir(parents=True)
            (clip / "preview.mp4").write_bytes(b"x")
            (clip / "episode.json").write_text("{not-json", encoding="utf-8")
            _write_heatmap(clip / "spatial_annotations" / "spatial_annotations.json")
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("episode.json", rec.invalid_reason or "")

    def test_unreadable_video_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = _make_complete_clip(root, dummy_video=True)
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertTrue(rec.invalid_reason)
            self.assertNotIn("incomplete", rec.invalid_reason or "")

    def test_broken_heatmaps_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = _make_complete_clip(root)
            _write_heatmap(
                clip / "spatial_annotations" / "spatial_annotations.json",
                n=10,
                grid=3,
            )
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("heatmaps", rec.invalid_reason or "")

    def test_too_few_frames_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = _make_complete_clip(root, n_frames=12)
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("reported_frames", rec.invalid_reason or "")
            self.assertIn("min_decoded_frames", rec.invalid_reason or "")


if __name__ == "__main__":
    unittest.main()
