"""Student adapters. Scoring and rendering never import a concrete class from here."""

from .registry import available_models, load_model

__all__ = ["available_models", "load_model"]
