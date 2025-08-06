from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

_qwen_profile: ModelProfile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return _qwen_profile
