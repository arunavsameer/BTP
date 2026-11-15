"""CollisionStudentV6: V2 backbone + Δ1 ResidualGatedFusion + CoordConv."""

from .losses import get_lambda_feat_v6, student_v6_loss
from .model import CollisionStudentV6, ResidualGatedFusionV6, n_params

__all__ = [
    "CollisionStudentV6",
    "ResidualGatedFusionV6",
    "get_lambda_feat_v6",
    "n_params",
    "student_v6_loss",
]
