"""Clutter-aware heatmap loss.

Run:  python tests/test_clutter_loss.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from collision_jepa.losses import collision_heatmap_loss, occupancy_bce  # noqa: E402


def test_ignore_band_skips_already_safe_cells():
    pred = torch.full((1, 5, 5), 0.05)
    target = torch.full((1, 5, 5), 0.04)
    loss = collision_heatmap_loss(pred, target, ignore_below=0.15, fa_weight=0.0)
    assert float(loss) < 1e-6, float(loss)


def test_hot_cell_is_not_ignored():
    pred = torch.zeros(1, 5, 5)
    target = torch.zeros(1, 5, 5)
    pred[0, 4, 2] = 0.2
    target[0, 4, 2] = 0.9
    loss = collision_heatmap_loss(pred, target, ignore_below=0.15, fa_weight=0.0)
    assert float(loss) > 0.01


def test_false_alarm_penalizes_caution_on_leaves():
    pred = torch.zeros(1, 5, 5)
    target = torch.zeros(1, 5, 5)
    pred[0, 0, 0] = 0.8  # leaf cell predicted as STOP-ish
    with_fa = collision_heatmap_loss(pred, target, ignore_below=0.15, fa_weight=1.0)
    no_fa = collision_heatmap_loss(pred, target, ignore_below=0.15, fa_weight=0.0)
    assert float(with_fa) > float(no_fa)


def test_occupancy_bce_empty_vs_hot():
    logits = torch.zeros(2, 5, 5)
    empty = torch.zeros(2, 5, 5)
    hot = torch.ones(2, 5, 5)
    assert float(occupancy_bce(logits, empty)) < float(occupancy_bce(logits, hot))


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} clutter-loss tests passed")


if __name__ == "__main__":
    _run()
