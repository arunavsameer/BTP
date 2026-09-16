"""Vectorized teacher clip packing.

Run:  python tests/test_pack_clips.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from collision_jepa.data.dataset import pack_teacher_clips, teacher_clip_indices  # noqa: E402


def test_pack_teacher_clips_shapes_and_gather():
    n, h, w = 40, 6, 6
    frames = np.arange(n * h * w * 3, dtype=np.uint8).reshape(n, h, w, 3)
    heatmaps = np.zeros((n, 5, 5), dtype=np.float32)
    heatmaps[21] = 1.0
    packed = pack_teacher_clips(frames, heatmaps, clip_frames=4, clip_stride=2, sample_stride=5)
    assert packed is not None
    clips, targets, ends = packed
    assert clips.dtype == np.uint8
    assert clips.shape[1:] == (4, 3, h, w)
    assert targets.shape[1:] == (5, 5)
    t_lo = (4 - 1) * 2
    expect_ends = np.arange(t_lo, n, 5)
    np.testing.assert_array_equal(ends, expect_ends)
    end = int(ends[0])
    idx = teacher_clip_indices(end, 4, 2)
    for t_i, fi in enumerate(idx):
        np.testing.assert_array_equal(clips[0, t_i], np.transpose(frames[fi], (2, 0, 1)))
    hit = np.where(ends == 21)[0]
    assert hit.size == 1
    assert float(targets[int(hit[0])].max()) == 1.0


def test_pack_teacher_clips_none_when_too_short():
    frames = np.zeros((4, 2, 2, 3), dtype=np.uint8)
    assert pack_teacher_clips(frames, None, clip_frames=8, clip_stride=2) is None


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} pack tests passed")


if __name__ == "__main__":
    _run()
