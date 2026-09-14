"""Horizontal flip must move RGB, H, and teacher Z together.

Run:  python tests/test_augment_flip.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.data.augment import ClipAugmentor  # noqa: E402


def test_flip_moves_heatmap_and_z_together():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, size=(3, 16, 16, 3), dtype=np.uint8)
    h_now = np.zeros((5, 5), dtype=np.float32)
    h_now[3, 0] = 0.9  # near-left
    z_plus = np.zeros((16, 5, 5), dtype=np.float32)
    z_plus[:, 3, 0] = 1.0  # same cell in latent space

    aug = ClipAugmentor(
        brightness=0.0, contrast=0.0, blur_prob=0.0, noise_std=0.0, flip_prob=1.0, seed=0
    )
    out_frames, (h_out,), (z_out,) = aug(frames, [h_now], [z_plus])

    assert np.allclose(h_out[3, 4], 0.9), h_out[3]
    assert np.allclose(h_out[3, 0], 0.0)
    assert np.allclose(z_out[:, 3, 4], 1.0)
    assert np.allclose(z_out[:, 3, 0], 0.0)
    assert np.allclose(out_frames[:, :, ::-1, :], frames)


def test_no_flip_keeps_left_cells():
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    hm = np.zeros((5, 5), dtype=np.float32)
    hm[4, 0] = 1.0
    z = np.zeros((4, 5, 5), dtype=np.float32)
    z[:, 4, 0] = 2.0
    aug = ClipAugmentor(
        brightness=0.0, contrast=0.0, blur_prob=0.0, noise_std=0.0, flip_prob=0.0, seed=1
    )
    _, (h_out,), (z_out,) = aug(frames, [hm], [z])
    assert np.allclose(h_out[4, 0], 1.0)
    assert np.allclose(z_out[:, 4, 0], 2.0)


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} flip tests passed")


if __name__ == "__main__":
    _run()
