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
from collision_jepa3.data.manifest import build_manifest, inspect_clip_dir


def _write_heatmap(path: Path, n: int = 450) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [{"frame_id": f"{i:06d}", "matrix": np.zeros((5, 5)).tolist()} for i in range(n)]
    path.write_text(json.dumps({"k": 5, "frames": frames}), encoding="utf-8")


class ManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = Config.load()

    def test_incomplete_clip_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "sample_0000_jaywalker" / "collision" / "bg_00"
            clip.mkdir(parents=True)
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertFalse(rec.valid)
            self.assertIn("incomplete", rec.invalid_reason or "")

    def test_complete_clip_is_accepted(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clip = root / "sample_0001_cube" / "collision" / "bg_00"
            clip.mkdir(parents=True)
            video = clip / "preview.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            wr = cv2.VideoWriter(str(video), fourcc, 30.0, (64, 48))
            for _ in range(450):
                wr.write(np.zeros((48, 64, 3), dtype=np.uint8))
            wr.release()
            (clip / "episode.json").write_text(
                json.dumps({"scenario": "cube", "diverge_frame": 365, "video": "preview.mp4", "biome": "street"}),
                encoding="utf-8",
            )
            _write_heatmap(clip / "spatial_annotations" / "spatial_annotations.json")
            rec = inspect_clip_dir(clip, root, self.cfg)
            self.assertTrue(rec.valid, rec.invalid_reason)
            self.assertEqual(rec.outcome_type, "collision")
            self.assertEqual(rec.background_id, "bg_00")
            self.assertEqual(rec.scenario_group_id, "sample_0001_cube")

    def test_build_manifest_skips_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bad = root / "sample_0000_x" / "collision" / "bg_00"
            bad.mkdir(parents=True)
            recs = build_manifest(self.cfg, root)
            self.assertEqual(len(recs), 1)
            self.assertFalse(recs[0].valid)


if __name__ == "__main__":
    unittest.main()
