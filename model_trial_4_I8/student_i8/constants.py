"""Inference constants for CollisionStudentV2 (checkpoint_epoch_22)."""

from __future__ import annotations

ARCH = "CollisionStudentV2"
SOURCE_CKPT = "checkpoint_epoch_22.pt"
STOP_THRESHOLD = 0.60
THREAT_THRESHOLD = 0.25
INPUT_SIZE = 256
T_FRAMES = 16
OFFSETS_16 = list(range(-90, 1, 6))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_LAYOUT = "[B, T=16, C=3, H=256, W=256] float in [0, 1]; ImageNet norm is inside the model"
BINARY_OVERLAY_REQUIRED = False
