"""RGB student distilled from Collision-JEPA teacher G. No V-JEPA at inference."""

from .student import CollisionStudent, MAX_PARAMS, load_frozen_teacher_heads

__all__ = ["CollisionStudent", "MAX_PARAMS", "load_frozen_teacher_heads"]
