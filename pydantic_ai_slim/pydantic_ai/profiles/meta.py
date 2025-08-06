from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return _model_profile

_model_profile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)
