"""Shape / frozen-decoder checks for the Trial-4 student. No Trial-3 writes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from student.model import StudentLatPairCF
from student.parent import decoder_state_hash, load_parent_decoder
from student.paths import EXPECTED_DECODER_HASH, parent_ckpt


class StudentTests(unittest.TestCase):
    def test_parent_decoder_hash_and_shapes(self):
        ckpt = parent_ckpt()
        if not ckpt.is_file():
            self.skipTest(f"parent ckpt missing: {ckpt}")
        decoder = load_parent_decoder(ckpt)
        self.assertEqual(decoder_state_hash(decoder), EXPECTED_DECODER_HASH)
        model = StudentLatPairCF(decoder)
        audit = model.param_audit()
        self.assertLess(audit["total"], 1_000_000)
        self.assertEqual(audit["decoder"], 47809)
        self.assertFalse(any(p.requires_grad for p in model.decoder.parameters()))
        clip = torch.rand(1, 16, 3, 256, 256)
        out = model(clip)
        self.assertEqual(tuple(out["F"].shape), (1, 128, 16, 16))
        self.assertEqual(tuple(out["Hhat_1s"].shape), (1, 5, 5))
        out["F"].sum().backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))


if __name__ == "__main__":
    unittest.main()
