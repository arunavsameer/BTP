from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.analysis.channel_icc import pair_w_b_icc, select_low_icc_channels


class ChannelIccTests(unittest.TestCase):
    def test_identical_pairs_high_icc(self) -> None:
        fa = np.random.randn(100, 32)
        fb = fa.copy()
        _, _, icc = pair_w_b_icc(fa, fb)
        self.assertTrue(np.allclose(icc, 1.0, atol=1e-5))

    def test_selection_respects_w_threshold(self) -> None:
        icc = np.linspace(0, 1, 40)
        w = np.concatenate([np.full(20, 0.1), np.full(20, 10.0)])
        sel = select_low_icc_channels(icc, w, low_frac=0.25)
        self.assertEqual(sel["n_eligible"], 20)
        self.assertEqual(sel["n_selected"], 5)


if __name__ == "__main__":
    unittest.main()
