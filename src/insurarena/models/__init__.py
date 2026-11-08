"""Single-model adapters: each is one fitted estimator with an explanation export."""
from insurarena.models.base import Explanation, ModelAdapter
from insurarena.models.registry import build_model, list_models

__all__ = ["Explanation", "ModelAdapter", "build_model", "list_models"]
