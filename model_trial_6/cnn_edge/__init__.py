"""CollisionStudentV2 student with Canny-on-RGB input."""

from .losses import cnn_edge_loss, student_v2_loss
from .model import CollisionCNNEdge, CollisionStudentV2, n_params

__all__ = [
    "CollisionStudentV2",
    "CollisionCNNEdge",
    "n_params",
    "student_v2_loss",
    "cnn_edge_loss",
]
