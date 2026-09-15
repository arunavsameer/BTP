"""Shape tests that do not download V-JEPA 2."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cvjepa.data.dataset import teacher_clip_indices  # noqa: E402
from cvjepa.losses import spatial_heatmap_weights  # noqa: E402
from cvjepa.models.decoder import HeatmapDecoder, HeatmapDelta  # noqa: E402
from cvjepa.models.lora import LoRALinear  # noqa: E402
from cvjepa.models.student import CollisionStudent  # noqa: E402
from cvjepa.models.teacher import CollisionTeacher, SpatialAdapter  # noqa: E402


class FakeBackbone(nn.Module):
    hidden_size = 48
    spatial = 4

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b = clip.shape[0]
        n = 2 * self.spatial * self.spatial
        return torch.randn(b, n, self.hidden_size, device=clip.device, dtype=torch.float32)


def test_teacher_clip_indices() -> None:
    idx = teacher_clip_indices(20, clip_frames=8, stride=2)
    assert idx[0] == 6 and idx[-1] == 20 and len(idx) == 8


def test_spatial_adapter() -> None:
    ad = SpatialAdapter(48, z_channels=8, feature_grid=5, pool_hidden=16)
    tokens = torch.randn(2, 2 * 4 * 4, 48)
    z = ad(tokens, spatial=4)
    assert z.shape == (2, 8, 5, 5)


def test_teacher_with_fake_backbone() -> None:
    m = CollisionTeacher("unused", z_channels=8, feature_grid=5, pool_hidden=16, backbone=FakeBackbone())
    clip = torch.rand(2, 8, 3, 32, 32)
    out = m(clip)
    assert out["z_t"].shape == (2, 8, 5, 5)
    assert out["h_plus_hat"].shape == (2, 5, 5)
    assert 0.0 <= float(out["h_plus_hat"].detach().min()) <= float(out["h_plus_hat"].detach().max()) <= 1.0


def test_student_five_frames() -> None:
    m = CollisionStudent(width=16, feature_grid=5, z_channels=8, n_frames=5, use_loom=True, copy_residual=True)
    x = torch.rand(1, 5, 3, 160, 160)
    pred = m.predict_future_heatmap(x)
    assert pred.shape == (1, 5, 5)
    heads = {f"decoder.{k}": v for k, v in m.decoder.state_dict().items()}
    heads.update({f"delta_head.{k}": v for k, v in m.delta_head.state_dict().items()})
    m.load_frozen_heads(heads)
    assert all(not p.requires_grad for p in m.decoder.parameters())


def test_lora_linear() -> None:
    lin = nn.Linear(6, 4)
    lora = LoRALinear(lin, rank=2, alpha=4.0)
    x = torch.randn(3, 6)
    y = lora(x)
    assert y.shape == (3, 4)
    assert not lin.weight.requires_grad
    assert lora.lora_A.requires_grad


def test_decoder() -> None:
    d = HeatmapDecoder(8, 16)
    z = torch.randn(2, 8, 5, 5)
    h = d(z)
    assert h.shape == (2, 5, 5)
    delta = HeatmapDelta(8, 16)(z)
    assert delta.min() >= -1.0 and delta.max() <= 1.0


def test_spatial_weights() -> None:
    w = spatial_heatmap_weights(5, 2.2, 1.5, 1.0, 0.65)
    assert abs(float(w.mean()) - 1.0) < 1e-5


if __name__ == "__main__":
    test_teacher_clip_indices()
    test_spatial_adapter()
    test_teacher_with_fake_backbone()
    test_student_five_frames()
    test_lora_linear()
    test_decoder()
    test_spatial_weights()
    print("ok")
