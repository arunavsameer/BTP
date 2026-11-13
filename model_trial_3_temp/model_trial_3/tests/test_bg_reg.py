from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.training.bg_regularization import (
    apply_pair_counterfactual_shrink,
    clean_anchor_drop_strength,
    pair_cf_ramp_strength,
    deterministic_channel_mask,
    pair_consistency_loss,
    token_channel_dropout,
)


class BgRegTests(unittest.TestCase):
    def test_pair_consistency_masked(self) -> None:
        ha = torch.rand(2, 5, 5)
        hb = ha + 0.1
        y = torch.zeros(2, 5, 5)
        mask = torch.zeros(2, dtype=torch.bool)
        w_mean = 0.05 + 4.0 * (y * y).mean()
        loss = pair_consistency_loss(ha, hb, y, mask, w_train_mean=float(w_mean))
        self.assertTrue(torch.isfinite(loss))

    def test_token_dropout_shape(self) -> None:
        tok = torch.ones(4, 8, 1408)
        ch = [0, 1, 2]
        out = token_channel_dropout(tok, ch, training=True, p_drop=0.5)
        self.assertEqual(out.shape, tok.shape)

    def test_deterministic_mask_stable(self) -> None:
        m1 = deterministic_channel_mask([0, 1, 2], c_dim=16, device=torch.device("cpu"), dtype=torch.float32, p_drop=0.25, key="a|b|1")
        m2 = deterministic_channel_mask([0, 1, 2], c_dim=16, device=torch.device("cpu"), dtype=torch.float32, p_drop=0.25, key="a|b|1")
        self.assertTrue(torch.allclose(m1, m2))

    def test_clean_anchor_strength_schedule(self) -> None:
        self.assertEqual(clean_anchor_drop_strength(0), 0.0)
        self.assertEqual(clean_anchor_drop_strength(1), 0.25)
        self.assertEqual(clean_anchor_drop_strength(4), 1.0)
        self.assertEqual(pair_cf_ramp_strength(5), 1.0)

    def test_pair_cf_preserves_pair_mean(self) -> None:
        rng = np.random.default_rng(0)
        U = torch.from_numpy(rng.standard_normal((4, 16)).astype(np.float32))
        Fa = torch.randn(2, 8, 16)
        Fb = Fa + torch.randn(2, 8, 16) * 0.1
        tok = torch.cat([Fa, Fb], dim=0)
        out = apply_pair_counterfactual_shrink(tok, U, lam=1.0)
        mid = 2
        savg = 0.5 * (tok[:mid].mean(1) + tok[mid:].mean(1))
        navg = 0.5 * (out[:mid].mean(1) + out[mid:].mean(1))
        self.assertTrue(torch.allclose(savg, navg, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
