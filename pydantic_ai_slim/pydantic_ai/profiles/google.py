from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """
    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Note: we need to remove `additionalProperties: False` since it is currently mishandled by Gemini
        additional_properties = schema.pop(
            'additionalProperties', None
        )  # don't pop yet so it's included in the warning
        if additional_properties:
            original_schema = {**schema, 'additionalProperties': additional_properties}
            warnings.warn(
                '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                f' Full schema: {self.schema}\n\n'
                f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                "If Google's APIs are updated to support this properly, please create an issue on the Pydantic AI GitHub"
                ' and we will fix this behavior.',
                UserWarning,
            )

        schema.pop('title', None)
        schema.pop('$schema', None)
        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]
        schema.pop('discriminator', None)
        schema.pop('examples', None)
        schema.pop('exclusiveMaximum', None)
        schema.pop('exclusiveMinimum', None)

        # Gemini only supports string enums, so we need to convert any enum values to strings.
        # Pydantic will take care of transforming the transformed string values to the correct type.
        enum = schema.get('enum')
        if enum:
            schema['type'] = 'string'
            # Use tuple to speed up "not in" checks in prefixItems unique_items, see below.
            # Here, simply use list comprehension, as it is already efficient:
            schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')
        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            # This gets hit when we have a discriminated union
            # Gemini returns an API error in this case even though it says in its error message it shouldn't...
            # Changing the oneOf to an anyOf prevents the API error and I think is functionally equivalent
            schema['anyOf'] = schema.pop('oneOf')

        if type_ == 'string' and (fmt := schema.pop('format', None)):
            description = schema.get('description')
            if description:
                schema['description'] = f'{description} (format: {fmt})'
            else:
                schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        if 'prefixItems' in schema:
            # prefixItems is not currently supported in Gemini, so we convert it to items for best compatibility
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')

            # OPTIMIZATION:
            # The line profile shows the "if item not in unique_items" loop is >90% of runtime.
            # Use a set for O(1) membership check, keep unique_items in the required order.
            # First, get a unique_items list and _set for fast membership check:
            unique_items = []
            seen = set()
            if items is not None:
                unique_items.append(items)
                # Hashable and unhashable: fallback to id()
                try:
                    seen.add(self._item_hash(items))
                except Exception:
                    pass

            for item in prefix_items:
                try:
                    key = self._item_hash(item)
                except Exception:
                    # As fallback, behave conservatively and do slow membership check
                    if item not in unique_items:
                        unique_items.append(item)
                    continue
                if key not in seen:
                    unique_items.append(item)
                    seen.add(key)

            if len(unique_items) > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif len(unique_items) == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema

    @staticmethod
    def _item_hash(item):
        # Try hash first (for hashable types: str, int, float, tuples, etc)
        try:
            return hash(item)
        except TypeError:
            # For dicts and lists, convert into a tuple of tuples
            if isinstance(item, dict):
                # Sort keys to ensure consistent output
                return tuple(sorted((k, GoogleJsonSchemaTransformer._item_hash(v)) for k, v in item.items()))
            elif isinstance(item, list):
                return tuple(GoogleJsonSchemaTransformer._item_hash(i) for i in item)
            else:
                # fallback to id to ensure stable identity for unhashable, non-dict/list objects
                return id(item)
