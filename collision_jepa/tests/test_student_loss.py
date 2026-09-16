"""Tests for balanced student loss.

Run:  python tests/test_student_loss.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from collision_jepa.losses import (  # noqa: E402
    balanced_term,
    direction_scores_torch,
    student_loss,
)
from collision_jepa.models.student import Student  # noqa: E402
from collision_jepa.warning import direction_scores  # noqa: E402

WARN = {
    "left_cols": [0, 1],
    "center_cols": [2],
    "right_cols": [3, 4],
    "row_weights": [0.4, 0.6, 0.8, 1.0, 1.0],
    "caution_threshold": 0.45,
    "stop_threshold": 0.75,
}


def test_balanced_term_scales_huge_term_down():
    ref = torch.tensor(1.0, requires_grad=True)
    term = torch.tensor(20.0, requires_grad=True)
    out = balanced_term(term, ref, fraction=0.15)
    assert abs(float(out) - 0.15) < 1e-5, float(out)
    out.backward()
    # Gradient stays on `term`, not the reference.
    assert term.grad is not None and float(term.grad) > 0
    assert ref.grad is None or float(ref.grad) == 0.0


def test_balanced_term_does_not_explode_tiny_term():
    ref = torch.tensor(1.0)
    term = torch.tensor(1e-8, requires_grad=True)
    out = balanced_term(term, ref, fraction=0.40, max_boost=5.0)
    # Would need 4e7x boost to hit 0.40; cap at 5x -> 5e-8.
    assert float(out) < 1e-6
    assert float(out) > 0


def test_direction_scores_match_numpy():
    rng = np.random.default_rng(0)
    hm = rng.random((5, 5)).astype(np.float32)
    np_scores = direction_scores(hm, WARN["row_weights"], WARN["left_cols"], WARN["center_cols"], WARN["right_cols"])
    t_scores = direction_scores_torch(
        torch.from_numpy(hm).unsqueeze(0),
        WARN["row_weights"],
        WARN["left_cols"],
        WARN["center_cols"],
        WARN["right_cols"],
    )
    for k in ("LEFT", "CENTER", "RIGHT"):
        assert abs(float(t_scores[k]) - np_scores[k]) < 1e-5, (k, float(t_scores[k]), np_scores[k])


def test_student_loss_aux_cannot_drown_heatmap():
    b, g = 4, 5
    pred = torch.rand(b, g, g, requires_grad=True)
    # Prediction is garbage on purpose so readout/STOP raw losses are large.
    target = torch.zeros(b, g, g)
    target[:, 4, 2] = 1.0
    out = {
        "h_plus_hat": pred,
        "h_now_hat": pred.detach() * 0.5,
        "h_plus_latent": torch.rand(b, g, g),
        "z_plus_hat": torch.randn(b, 16, g, g),
    }
    batch = {
        "h_future": target,
        "h_now": target * 0.5,
        "z_plus": torch.randn(b, 16, g, g) * 50.0,
    }
    loss, parts = student_loss(
        out,
        batch,
        WARN,
        readout_frac=0.40,
        stop_frac=0.15,
        jepa_lambda=1.0,
        jepa_frac=0.10,
        latent_heatmap_frac=0.30,
        now_heatmap_weight=0.5,
    )
    h = float(parts["h_plus"])
    assert float(parts["readout"]) <= 0.40 * h + 1e-5, (float(parts["readout"]), h)
    assert float(parts["stop"]) <= 0.15 * h + 1e-5, (float(parts["stop"]), h)
    assert float(parts["jepa"]) <= 0.10 * h + 1e-5, (float(parts["jepa"]), h)
    assert float(parts["latent"]) <= 0.30 * h + 1e-5, (float(parts["latent"]), h)
    assert float(parts["readout_raw"]) > float(parts["readout"])
    loss.backward()
    assert pred.grad is not None


def test_delta_h_is_driven_by_zhat():
    """Deployed Ĥ+ must change when the JEPA predictor's Zhat+ changes."""
    m = Student(width=32, feature_grid=5, z_channels=16)
    m.eval()
    frames = torch.rand(2, 3, 3, 128, 128)
    with torch.no_grad():
        out = m(frames)
        z_t, z_plus = out["z_t"], out["z_plus_hat"]
        h_a = out["h_plus_hat"]
        h_b = torch.clamp(out["h_now_hat"] + m.delta_h((z_plus + 1.0) - z_t), 0.0, 1.0)
    assert not torch.allclose(h_a, h_b, atol=1e-4)


def test_future_path_stops_grad_through_current_map():
    m = Student(width=32, feature_grid=5, z_channels=16)
    frames = torch.rand(2, 3, 3, 128, 128)
    out = m(frames)
    out["h_plus_hat"].sum().backward()
    for p in m.decoder.parameters():
        if p.grad is not None:
            assert torch.count_nonzero(p.grad).item() == 0


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} student-loss tests passed")


if __name__ == "__main__":
    _run()
