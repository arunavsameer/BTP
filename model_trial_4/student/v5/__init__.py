"""CollisionStudentV5: V4 architecture, V5 loss + decoder unfreeze schedule."""

from .losses import get_lambda_feat_v5, student_v5_loss
from .model import CollisionStudentV5, n_params

__all__ = ["CollisionStudentV5", "get_lambda_feat_v5", "n_params", "student_v5_loss"]
