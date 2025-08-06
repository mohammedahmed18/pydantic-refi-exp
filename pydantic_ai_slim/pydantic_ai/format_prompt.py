from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)


def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    el = _ToXml(item_tag=item_tag, none_str=none_str).to_xml(obj, root_tag)
    if root_tag is None and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        item_tag = self.item_tag
        none_str = self.none_str
        et_Element = ElementTree.Element

        # Fastest: primitives first, then resolve complex types
        if value is None:
            element = et_Element(item_tag if tag is None else tag)
            element.text = none_str
            return element

        if isinstance(value, str):
            element = et_Element(item_tag if tag is None else tag)
            element.text = value
            return element

        if isinstance(value, (bytes, bytearray)):
            element = et_Element(item_tag if tag is None else tag)
            element.text = value.decode(errors='ignore')
            return element

        if isinstance(value, (bool, int, float)):
            element = et_Element(item_tag if tag is None else tag)
            element.text = str(value)
            return element

        if isinstance(value, date):
            element = et_Element(item_tag if tag is None else tag)
            element.text = value.isoformat()
            return element

        # Check dataclass early, it's faster than Mapping for dataclasses
        if is_dataclass(value) and not isinstance(value, type):
            cls_name = value.__class__.__name__
            # Use class name as tag if not provided
            tag_ = cls_name if tag is None else tag
            element = et_Element(tag_)
            # Use __dict__ directly for flat dataclasses for speed, else fallback to asdict
            try:
                dc_dict = value.__dict__
            except AttributeError:
                dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
            return element

        # Next, handle BaseModel (Pydantic)
        if isinstance(value, BaseModel):
            tag_ = value.__class__.__name__ if tag is None else tag
            element = et_Element(tag_)
            # model_dump returns dict; skip any exclude_unset complexity for speed
            self._mapping_to_xml(element, value.model_dump(mode='python'))
            return element

        # Mapping case: could be dict, Mapping, etc.
        if isinstance(value, Mapping):
            element = et_Element(item_tag if tag is None else tag)
            self._mapping_to_xml(element, value)
            return element

        # Iterable
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            element = et_Element(item_tag if tag is None else tag)
            # Avoid attribute lookup inside loop and cache helpers
            append = element.append
            to_xml = self.to_xml
            for item in value:
                append(to_xml(item, None))
            return element

        # If not handled above, error out
        raise TypeError(f'Unsupported type for XML formatting: {type(value)}')

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    # Avoid repeated lookups for indent/tostring
    indent_func = ElementTree.indent if indent is not None else None
    tostring = ElementTree.tostring
    for sub_element in root:
        if indent_func is not None:
            indent_func(sub_element, space=indent)
        yield tostring(sub_element, encoding='unicode')
