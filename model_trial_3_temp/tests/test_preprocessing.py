from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.data.preprocessing import (
    PreprocessSignature,
    letterbox_rgb,
    preprocess_frame,
    uint8_to_nchw01,
)


class PreprocessTests(unittest.TestCase):
    def test_letterbox_keeps_aspect(self) -> None:
        frame = np.zeros((108, 192, 3), dtype=np.uint8)
        frame[:, :10] = (255, 0, 0)  # left stripe
        frame[:, -10:] = (0, 0, 255)  # right stripe
        out = letterbox_rgb(frame, 64, pad_value=0)
        self.assertEqual(out.shape, (64, 64, 3))
        # red should still be on the left half, blue on the right
        self.assertGreater(int(out[:, :20, 0].sum()), int(out[:, -20:, 0].sum()))
        self.assertGreater(int(out[:, -20:, 2].sum()), int(out[:, :20, 2].sum()))

    def test_rgb_order_not_swapped(self) -> None:
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        frame[..., 0] = 200
        out = letterbox_rgb(frame, 32)
        self.assertGreater(int(out[..., 0].mean()), int(out[..., 2].mean()))

    def test_uint8_to_unit_interval(self) -> None:
        clip = np.zeros((4, 8, 8, 3), dtype=np.uint8)
        clip[0, 0, 0] = (255, 128, 0)
        x = uint8_to_nchw01(clip)
        self.assertEqual(x.shape, (4, 3, 8, 8))
        self.assertAlmostEqual(float(x[0, 0, 0, 0]), 1.0, places=5)
        self.assertTrue(float(x.min()) >= 0.0 and float(x.max()) <= 1.0)

    def test_deterministic(self) -> None:
        sig = PreprocessSignature("letterbox", 32, 0, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), True)
        frame = np.random.RandomState(0).randint(0, 255, (45, 80, 3), dtype=np.uint8)
        a = preprocess_frame(frame, sig)
        b = preprocess_frame(frame, sig)
        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
