from __future__ import annotations as _annotations

from pydantic_ai.messages import TextPart, ThinkingPart

START_THINK_TAG = '<think>'
END_THINK_TAG = '</think>'


def split_content_into_text_and_thinking(content: str) -> list[ThinkingPart | TextPart]:
    """Split a string into text and thinking parts.

    Some models don't return the thinking part as a separate part, but rather as a tag in the content.
    This function splits the content into text and thinking parts.

    We use the `<think>` tag because that's how Groq uses it in the `raw` format, so instead of using `<Thinking>` or
    something else, we just match the tag to make it easier for other models that don't support the `ThinkingPart`.
    """
    parts: list[ThinkingPart | TextPart] = []
    start_len = len(START_THINK_TAG)
    end_len = len(END_THINK_TAG)

    pos = 0
    while True:
        start_index = content.find(START_THINK_TAG, pos)
        if start_index == -1:
            break
        if start_index > pos:
            parts.append(TextPart(content=content[pos:start_index]))
        think_start = start_index + start_len
        end_index = content.find(END_THINK_TAG, think_start)
        if end_index != -1:
            parts.append(ThinkingPart(content=content[think_start:end_index]))
            pos = end_index + end_len
        else:
            parts.append(TextPart(content=content[think_start:]))
            return parts
    if pos < len(content):
        parts.append(TextPart(content=content[pos:]))
    return parts
