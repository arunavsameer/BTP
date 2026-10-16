from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa3.config import Config
from collision_jepa3.data.sampling import FRAME_OFFSETS, input_indices


class NoFutureLeakageTests(unittest.TestCase):
    def test_no_offset_after_anchor(self) -> None:
        self.assertTrue(all(o <= 0 for o in FRAME_OFFSETS))

    def test_config_rejects_future_offsets(self) -> None:
        cfg = Config.load()
        cfg.validate()
        for t in (90, 200, 389, 419):
            idx = input_indices(t, cfg.get("sampling.frame_offsets"))
            self.assertTrue(all(i <= t for i in idx))
            self.assertEqual(idx[-1], t)

    def test_input_indices_raises_on_future(self) -> None:
        with self.assertRaises(ValueError):
            input_indices(10, [-4, 2])


if __name__ == "__main__":
    unittest.main()
