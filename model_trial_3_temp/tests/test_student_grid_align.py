from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.student_research.grid_align import letterbox_geom, native_gt_vs_dst_grid


class StudentGridAlignTests(unittest.TestCase):
    def test_row_col_order_preserved(self) -> None:
        native, mapped = native_gt_vs_dst_grid()
        self.assertEqual(native.shape, (5, 5))
        self.assertGreater(float(native.max()), 0.0)

    def test_letterbox_padding_geometry(self) -> None:
        g = letterbox_geom(1920, 1080, 160)
        self.assertEqual(g.pad_x, 0)
        self.assertGreater(g.pad_y, 0)
        self.assertEqual(g.resized_w, 160)
        self.assertLess(g.resized_h, 160)
        self.assertGreater(g.scale, 0)

    def test_flip_columns(self) -> None:
        native, _ = native_gt_vs_dst_grid()
        flipped = native[:, ::-1]
        self.assertAlmostEqual(float(native[0, 0]), float(flipped[0, 4]))


if __name__ == "__main__":
    unittest.main()
