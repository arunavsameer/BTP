"""Shape/smoke tests for the models. Requires torch.

Run:  python tests/test_shapes.py   (or: pytest tests/test_shapes.py)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from collision_jepa.models.baseline import Stage0Model  # noqa: E402
from collision_jepa.models.decoder import HeatmapDecoder  # noqa: E402
from collision_jepa.models.student import Student  # noqa: E402
from collision_jepa.models.tiny_cnn import TinyCNN, count_params  # noqa: E402

B, N, C, H, W = 2, 3, 3, 128, 128
GRID, ZC = 5, 16


def test_tiny_cnn_output_grid():
    cnn = TinyCNN(width=32, feature_grid=GRID)
    x = torch.randn(B, 3, H, W)
    y = cnn(x)
    assert y.shape == (B, cnn.feat_channels, GRID, GRID), y.shape


def test_decoder_shape_and_range():
    dec = HeatmapDecoder(z_channels=ZC)
    z = torch.randn(B, ZC, GRID, GRID)
    h = dec(z)
    assert h.shape == (B, GRID, GRID)
    assert float(h.min()) >= 0.0 and float(h.max()) <= 1.0


def test_stage0_forward():
    m = Stage0Model(width=32, feature_grid=GRID)
    frames = torch.rand(B, N, C, H, W)
    h = m(frames)
    assert h.shape == (B, GRID, GRID)
    assert float(h.min()) >= 0.0 and float(h.max()) <= 1.0


def test_student_forward_and_params():
    m = Student(width=32, feature_grid=GRID, z_channels=ZC)
    frames = torch.rand(B, N, C, H, W)
    out = m(frames)
    assert out["z_t"].shape == (B, ZC, GRID, GRID)
    assert out["z_plus_hat"].shape == (B, ZC, GRID, GRID)
    assert out["h_plus_hat"].shape == (B, GRID, GRID)
    h = m.predict_future_heatmap(frames)
    assert h.shape == (B, GRID, GRID)
    n = count_params(m)
    assert n < 3_000_000, f"student too large: {n:,} params"
    print(f"student params: {n:,}")


def test_student_frozen_decoder():
    m = Student(width=32, feature_grid=GRID, z_channels=ZC)
    dec = HeatmapDecoder(z_channels=ZC)
    m.load_frozen_decoder(dec.state_dict())
    assert all(not p.requires_grad for p in m.decoder.parameters())


def _run():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"all {len(fns)} shape tests passed")


if __name__ == "__main__":
    _run()
