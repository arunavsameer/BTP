"""RGB student: owned heatmap decoder + ordinal severity head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from collision_jepa3.student_research.cnn import MAX_TOTAL_PARAMS, StudentRGBNet, assert_param_budget
from collision_jepa3.student_research.fdec_student import (
    FDEC_FRAME_OFFSETS,
    backbone_state_keys,
    base_concat_config,
    extract_backbone_state,
    load_backbone_state,
)


class OrdinalSeverityHead(nn.Module):
  def __init__(self, fine_channels: int = 128, grid: int = 5):
    super().__init__()
    self.fine_channels = int(fine_channels)
    self.grid = int(grid)
    self.mlp = nn.Sequential(
      nn.Linear(fine_channels * 2 + grid * grid, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 1),
    )
    self.theta_warn = nn.Parameter(torch.tensor(0.0))
    self.delta = nn.Parameter(torch.tensor(-1.0))

  def forward(self, fine: torch.Tensor, heatmap: torch.Tensor) -> dict[str, torch.Tensor]:
    gap = fine.mean(dim=(2, 3))
    gmp = fine.amax(dim=(2, 3))
    h_flat = heatmap.detach().reshape(heatmap.shape[0], -1)
    x = torch.cat([gap, gmp, h_flat], dim=1)
    s = self.mlp(x).squeeze(1)
    theta_stop = self.theta_warn + F.softplus(self.delta)
    q_warn = torch.sigmoid(s - self.theta_warn)
    q_stop = torch.sigmoid(s - theta_stop)
    p_safe = 1.0 - q_warn
    p_threat = (q_warn - q_stop).clamp_min(0.0)
    p_stop = q_stop
    return {
      "s": s,
      "theta_warn": self.theta_warn,
      "theta_stop": theta_stop,
      "q_warn": q_warn,
      "q_stop": q_stop,
      "P_SAFE": p_safe,
      "P_THREAT": p_threat,
      "P_STOP": p_stop,
    }


class StudentFinal(nn.Module):
  """Shared Base CNN backbone + owned decoder + ordinal severity."""

  def __init__(self, *, n_frames: int = len(FDEC_FRAME_OFFSETS), input_size: int = 160):
    super().__init__()
    arch = base_concat_config()
    self.rgb = StudentRGBNet(arch, n_frames=n_frames, input_size=input_size)
    self.severity = OrdinalSeverityHead(fine_channels=arch.fine_channels, grid=arch.out_grid)

  def forward(self, clip: torch.Tensor) -> dict[str, torch.Tensor]:
    fine = self.rgb._spatial_feats(clip)
    h, logits = self.rgb.decoder.forward_with_logits(fine)
    sev = self.severity(fine, h)
    out = {
      "F": fine,
      "Hhat_1s": h,
      "Hhat_1s_logits": logits,
      **sev,
    }
    return out

  def load_fresh_backbone(self, backbone_state: dict[str, torch.Tensor]) -> None:
    load_backbone_state(self.rgb, backbone_state)

  def backbone_state(self) -> dict[str, torch.Tensor]:
    return extract_backbone_state(self.rgb)

  def param_audit(self) -> dict[str, Any]:
    total = sum(p.numel() for p in self.parameters())
    dec = sum(p.numel() for p in self.rgb.decoder.parameters())
    bb = total - dec - sum(p.numel() for p in self.severity.parameters())
    sev = sum(p.numel() for p in self.severity.parameters())
    assert_param_budget(self, MAX_TOTAL_PARAMS)
    return {
      "variant": "student_final_from_scratch",
      "total_params": total,
      "backbone_params": bb,
      "decoder_params": dec,
      "severity_params": sev,
      "backbone_keys": list(backbone_state_keys()),
    }
