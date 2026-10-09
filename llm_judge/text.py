"""Shared text extraction helpers."""

from __future__ import annotations

import re

_THINK_TAG_RE = re.compile(r"</?think\b[^>]*>", re.IGNORECASE)


def strip_think_content(text) -> str:
    """Remove <think>...</think> blocks (DeepSeek/Baichuan reasoning tags)."""
    if text is None:
        return ""

    text = str(text)

    # Handle orphan </think> (no opening <think>)
    last_orphan_close = None
    depth = 0
    for match in _THINK_TAG_RE.finditer(text):
        if match.group(0).lower().startswith("</"):
            if depth > 0:
                depth -= 1
            else:
                last_orphan_close = match
        else:
            depth += 1

    if last_orphan_close is not None:
        return strip_think_content(text[last_orphan_close.end() :])

    # Normal paired <think>...</think> removal
    parts = []
    cursor = 0
    depth = 0

    for match in _THINK_TAG_RE.finditer(text):
        tag = match.group(0).lower()
        is_close = tag.startswith("</")

        if depth == 0:
            parts.append(text[cursor : match.start()])

        if is_close:
            if depth > 0:
                depth -= 1
        else:
            depth += 1

        cursor = match.end()

    if depth == 0:
        parts.append(text[cursor:])

    return "".join(parts).strip()


def extract_completion_text(completion_item) -> str:
    """Extract text content from various completion formats."""
    if isinstance(completion_item, str):
        raw = completion_item
    elif isinstance(completion_item, dict):
        raw = str(completion_item.get("content", ""))
    elif isinstance(completion_item, list):
        chunks = []
        for item in completion_item:
            if isinstance(item, dict):
                chunks.append(str(item.get("content", "")))
            else:
                chunks.append(str(item))
        raw = "\n".join(chunks)
    else:
        raw = str(completion_item)

    return strip_think_content(raw)


__all__ = ["strip_think_content", "extract_completion_text"]
