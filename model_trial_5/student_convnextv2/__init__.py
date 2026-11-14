"""3D ConvNeXt-V2 student with submanifold sparse convolutions."""

from .model import CollisionStudentConvNeXtV2, VARIANTS, n_params
from .losses import convnext_student_loss

__all__ = [
    "CollisionStudentConvNeXtV2",
    "VARIANTS",
    "n_params",
    "convnext_student_loss",
]
