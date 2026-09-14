"""Unit tests for region shuffle invertibility and 180° rotation rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rs_jepa.data.shuffle import (  # noqa: E402
    apply_shuffle,
    apply_shuffle_pair,
    permute_grid,
    rotate_mask_from_perm,
    sample_perm,
    unpermute_grid,
    unpermute_map,
)
from rs_jepa.data.gpu_preprocess import prepare_batch  # noqa: E402
from rs_jepa.data.video import upsample_heatmap  # noqa: E402
from rs_jepa.models.student import RSJEPA  # noqa: E402
from rs_jepa.models.tiny_cnn import count_params  # noqa: E402


def test_rotate_rule() -> None:
    grid = 5
    perm = torch.arange(25).view(1, 25)
    perm[0, 24] = 0
    perm[0, 0] = 24
    rot = rotate_mask_from_perm(perm, grid)[0]
    assert bool(rot[24]), "top-left moved to bottom-right must rotate"
    assert bool(rot[0]), "bottom-right moved to top-left must rotate"

    perm = torch.arange(25).view(1, 25)
    perm[0, 14] = 10
    perm[0, 10] = 14
    rot = rotate_mask_from_perm(perm, grid)[0]
    assert bool(rot[14])
    assert bool(rot[10])

    perm = torch.arange(25).view(1, 25)
    perm[0, 1] = 0
    perm[0, 0] = 1
    rot = rotate_mask_from_perm(perm, grid)[0]
    assert not bool(rot[1]), "same-side adjacent move should not rotate"


def test_heatmap_roundtrip() -> None:
    grid = 5
    hm = torch.arange(25, dtype=torch.float32).view(1, 5, 5)
    perm = torch.randperm(25).view(1, 25)
    shuf = permute_grid(hm, perm, grid)
    back = unpermute_grid(shuf, perm, grid)
    assert torch.allclose(back, hm)


def test_unpermute_map_roundtrip() -> None:
    z = torch.randn(2, 8, 5, 5)
    perm = torch.stack([torch.randperm(25), torch.randperm(25)])
    idx = perm.unsqueeze(1).expand(-1, 8, -1)
    shuf = z.reshape(2, 8, 25).gather(2, idx).reshape(2, 8, 5, 5)
    rec = unpermute_map(shuf, perm, 5)
    assert torch.allclose(rec, z)


def test_sample_perm_identity() -> None:
    perm, rot = sample_perm(8, 5, 0.0, "cpu")
    assert torch.equal(perm, torch.arange(25).expand(8, 25))
    assert not bool(rot.any())


def test_frame_shuffle_roundtrip_identity() -> None:
    frames = torch.randn(2, 3, 3, 160, 160)
    perm = torch.arange(25).expand(2, 25).contiguous()
    rot = torch.zeros(2, 25, dtype=torch.bool)
    out = apply_shuffle(frames, perm, rot, 5)
    assert torch.allclose(out, frames, atol=1e-6)


def test_shuffle_pair_matches_two_calls() -> None:
    now = torch.randn(2, 3, 3, 160, 160)
    fut = torch.randn(2, 3, 3, 160, 160)
    perm = torch.stack([torch.randperm(25), torch.randperm(25)])
    rot = rotate_mask_from_perm(perm, 5)
    a, b = apply_shuffle_pair(now, fut, perm, rot, 5)
    assert torch.allclose(a, apply_shuffle(now, perm, rot, 5), atol=1e-5)
    assert torch.allclose(b, apply_shuffle(fut, perm, rot, 5), atol=1e-5)


def test_deshuffle_recovers_painted_cell() -> None:
    """A unique colour in cell 0, shuffled to slot k, deshuffled heatmap matches."""
    grid = 5
    frames = torch.zeros(1, 3, 3, 160, 160)
    ch = 160 // grid
    frames[:, :, :, 0:ch, 0:ch] = 0.7
    perm = torch.arange(25).view(1, 25)
    perm[0, 24] = 0
    perm[0, 0] = 24
    rot = rotate_mask_from_perm(perm, grid)
    shuf = apply_shuffle(frames, perm, rot, grid)
    br = shuf[:, :, :, 4 * ch : 5 * ch, 4 * ch : 5 * ch]
    assert float(br.mean()) > 0.5


def test_nearest_upsample_keeps_peak() -> None:
    hm = np.zeros((1, 3, 3), dtype=np.float32)
    hm[0, 0, 0] = 1.0
    out = upsample_heatmap(hm, grid=5, mode="nearest")
    assert out.shape == (1, 5, 5)
    assert float(out[0, 0, 0]) == 1.0
    assert float(out[0, 2, 2]) == 0.0
    native = upsample_heatmap(hm, grid=3, mode="nearest")
    assert native.shape == (1, 3, 3)
    assert float(native[0, 0, 0]) == 1.0


def test_model_shapes() -> None:
    m = RSJEPA(width=16, feature_grid=5, z_channels=8, n_frames=3, use_loom=True)
    x = torch.rand(2, 3, 3, 160, 160)
    y = torch.rand(2, 3, 3, 160, 160)
    out = m(x, y)
    assert out["h_plus_hat"].shape == (2, 5, 5)
    assert out["h_mid_hat"].shape == (2, 5, 5)
    assert out["z_plus"].shape == (2, 8, 5, 5)
    assert out["z_plus_hat"].shape == (2, 8, 5, 5)
    pred = m.predict_future_heatmap(x)
    assert pred.shape == (2, 5, 5)
    assert 0.0 <= float(pred.min()) and float(pred.max()) <= 1.0
    n = count_params(m)
    assert n < 2_000_000, n

    perm = torch.stack([torch.randperm(25), torch.randperm(25)])
    out_p = m(x, y, perm=perm)
    assert out_p["h_plus_hat"].shape == (2, 5, 5)


def test_model_four_frames() -> None:
    m = RSJEPA(width=16, feature_grid=5, z_channels=8, n_frames=4, use_loom=True)
    x = torch.rand(1, 4, 3, 160, 160)
    pred = m.predict_future_heatmap(x)
    assert pred.shape == (1, 5, 5)
    assert count_params(m) < 2_500_000


def test_prepare_batch_uint8() -> None:
    batch = {
        "frames": torch.randint(0, 255, (2, 4, 160, 160, 3), dtype=torch.uint8),
        "frames_future": torch.randint(0, 255, (2, 4, 160, 160, 3), dtype=torch.uint8),
        "h_now": torch.rand(2, 5, 5),
        "h_future": torch.rand(2, 5, 5),
        "h_mid": torch.rand(2, 5, 5),
        "ep_idx": torch.zeros(2, dtype=torch.long),
        "t": torch.zeros(2, dtype=torch.long),
    }
    out = prepare_batch(batch, "cpu", augment=True)
    assert out["frames"].shape == (2, 4, 3, 160, 160)
    assert out["frames"].dtype == torch.float32
    assert 0.0 <= float(out["frames"].min()) and float(out["frames"].max()) <= 1.0
    out2 = prepare_batch(batch, "cpu", augment=False)
    assert torch.allclose(out2["frames"], batch["frames"].permute(0, 1, 4, 2, 3).float() / 255, atol=1e-5)


def test_grid3_shuffle_and_model() -> None:
    grid = 3
    n = grid * grid
    perm, rot = sample_perm(4, grid, 0.0, "cpu")
    assert torch.equal(perm, torch.arange(n).expand(4, n))
    frames = torch.zeros(1, 3, 3, 160, 160)
    ch = (160 // grid) * grid // grid
    frames[:, :, :, 0:ch, 0:ch] = 0.7
    perm = torch.arange(n).view(1, n)
    perm[0, 8] = 0
    perm[0, 0] = 8
    rot = rotate_mask_from_perm(perm, grid)
    assert bool(rot[0, 8]) and bool(rot[0, 0])
    shuf = apply_shuffle(frames, perm, rot, grid)
    br = shuf[:, :, :, 2 * ch : 3 * ch, 2 * ch : 3 * ch]
    assert float(br.mean()) > 0.5
    m = RSJEPA(width=16, feature_grid=3, z_channels=8, n_frames=4, use_loom=True)
    x = torch.rand(2, 4, 3, 160, 160)
    pred = m.predict_future_heatmap(x)
    assert pred.shape == (2, 3, 3)
    out = m(x, x, perm=torch.stack([torch.randperm(n), torch.randperm(n)]))
    assert out["h_plus_hat"].shape == (2, 3, 3)


def test_copy_residual_shapes() -> None:
    m = RSJEPA(width=16, feature_grid=3, z_channels=8, n_frames=4, use_loom=True, copy_residual=True)
    x = torch.rand(2, 4, 3, 160, 160)
    out = m(x, x)
    assert out["delta"].shape == (2, 3, 3)
    assert out["h_plus_hat"].shape == (2, 3, 3)
    assert 0.0 <= float(out["h_plus_hat"].detach().min())
    assert float(out["h_plus_hat"].detach().max()) <= 1.0
    pred = m.predict_future_heatmap(x)
    assert pred.shape == (2, 3, 3)


if __name__ == "__main__":
    test_rotate_rule()
    test_heatmap_roundtrip()
    test_unpermute_map_roundtrip()
    test_sample_perm_identity()
    test_frame_shuffle_roundtrip_identity()
    test_shuffle_pair_matches_two_calls()
    test_deshuffle_recovers_painted_cell()
    test_nearest_upsample_keeps_peak()
    test_model_shapes()
    test_model_four_frames()
    test_prepare_batch_uint8()
    test_grid3_shuffle_and_model()
    test_copy_residual_shapes()
    print("ok")
