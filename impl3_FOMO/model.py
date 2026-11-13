"""FOMO head on a frozen MobileNetV2, cut where the map is stride 8.

Each cell is a classifier: background, or one walkway class. The positive
cell during training is the one that contains the object's centroid.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _stride8_backbone(input_size: int, torch_home: str) -> tuple[nn.Sequential, int]:
    os.environ["TORCH_HOME"] = torch_home
    net = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    kept = []
    probe = torch.zeros(1, 3, input_size, input_size)
    channels = 0
    with torch.no_grad():
        for layer in net.features:
            probe = layer(probe)
            kept.append(layer)
            channels = int(probe.shape[1])
            if int(probe.shape[-1]) == input_size // 8:
                break
    backbone = nn.Sequential(*kept)
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone.eval()
    return backbone, channels


class Fomo(nn.Module):
    def __init__(self, num_classes: int, input_size: int = 192, torch_home: str = "cache"):
        super().__init__()
        self.input_size = int(input_size)
        self.num_classes = int(num_classes)  # objects only; logit 0 is background
        self.backbone, channels = _stride8_backbone(self.input_size, torch_home)
        # Class logits, then log(width) and log(height) in original-frame pixels.
        self.head = nn.Conv2d(channels, self.num_classes + 1 + 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # Start predicting background, so empty street cells stay quiet.
        self.head.bias.data[0] = 2.0
        self.head.bias.data[-2] = math.log(48.0)
        self.head.bias.data[-1] = math.log(96.0)
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbone statistics stay in eval even while the head trains.
        self.backbone.eval()
        return self

    def normalize(self, images: torch.Tensor) -> torch.Tensor:
        """images are float NCHW in 0..1."""
        return (images - self.mean) / self.std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images uint8-scaled float 0..1, NCHW. Returns logits [N, 1+C, G, G]."""
        features = self.backbone(self.normalize(images))
        return self.head(features)

    def features_from_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.backbone(self.normalize(images))
