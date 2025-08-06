from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model."""
    return _model_profile

_model_profile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)
