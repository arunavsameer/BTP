from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.training.pair_signals import (
    DIR_ALPHAS,
    is_canonical_background,
    pcgrad_combine,
    per_sample_bce,
)


class PairSignalTests(unittest.TestCase):
    def test_canonical_bg00_only(self) -> None:
        self.assertTrue(is_canonical_background("bg_00", "s/collision/bg_00"))
        self.assertFalse(is_canonical_background("bg_01", "s/collision/bg_01"))

    def test_interpolant_is_convex(self) -> None:
        fa = torch.zeros(1, 2, 2, 2)
        fb = torch.ones(1, 2, 2, 2)
        for a in DIR_ALPHAS:
            f = fa + a * (fb - fa)
            self.assertTrue(torch.allclose(f, torch.full_like(fa, a)))
        self.assertTrue(torch.allclose(fa + 0.5 * (fb - fa), 0.5 * (fa + fb)))

    def test_per_sample_bce_ranks_harder_higher(self) -> None:
        y = torch.ones(2, 3, 3)
        easy = torch.full((2, 3, 3), 4.0)
        hard = torch.full((2, 3, 3), -4.0)
        mask = torch.zeros(2, dtype=torch.bool)
        se = per_sample_bce(easy, y, mask)
        sh = per_sample_bce(hard, y, mask)
        self.assertTrue(torch.all(sh > se))

    def test_pcgrad_agrees_when_cosine_positive(self) -> None:
        ga = [torch.tensor([1.0, 0.0]), torch.tensor([0.5])]
        gb = [torch.tensor([2.0, 0.0]), torch.tensor([1.0])]
        mixed, cos, conflict = pcgrad_combine(ga, gb)
        self.assertFalse(conflict)
        self.assertGreater(cos, 0.0)
        self.assertTrue(torch.allclose(mixed[0], torch.tensor([1.5, 0.0])))

    def test_pcgrad_projects_when_cosine_negative(self) -> None:
        ga = [torch.tensor([1.0, 0.0])]
        gb = [torch.tensor([-1.0, 0.0])]
        mixed, cos, conflict = pcgrad_combine(ga, gb)
        self.assertTrue(conflict)
        self.assertLess(cos, 0.0)
        # After removing the opposing component both grads are 0, average is 0.
        self.assertTrue(torch.allclose(mixed[0], torch.zeros(2), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
