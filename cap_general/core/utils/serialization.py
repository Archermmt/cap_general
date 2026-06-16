"""Serialization helpers for API and MCP responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def to_json_safe(value: Any) -> Any:
    """Recursively convert array-like values into JSON/MCP friendly values."""
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_json_safe(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, str | bytes):
        return to_json_safe(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, float | int):
        return float(value)
    return value


def summarize_value(value: Any, *, depth: int = 0) -> str:
    """Return a compact, log-safe representation of nested values."""
    max_depth = 2
    max_items = 5
    max_text = 160

    if value is None or isinstance(value, bool | int | float):
        return repr(value)
    if depth >= max_depth:
        return f"<{type(value).__name__}>"
    if isinstance(value, str):
        text = value.replace("\n", "\\n")
        if len(text) > max_text:
            text = f"{text[:max_text]}...<truncated {len(text) - max_text} chars>"
        return repr(text)
    if isinstance(value, Path):
        return repr(str(value))
    if isinstance(value, dict):
        items = list(value.items())
        parts = [
            f"{key!r}: {summarize_value(item_value, depth=depth + 1)}"
            for key, item_value in items[:max_items]
        ]
        if len(items) > max_items:
            parts.append(f"... +{len(items) - max_items} more")
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, tuple):
        parts = [summarize_value(item, depth=depth + 1) for item in value[:max_items]]
        if len(value) > max_items:
            parts.append(f"... +{len(value) - max_items} more")
        trailing = "," if len(value) == 1 else ""
        return "(" + ", ".join(parts) + trailing + ")"
    if isinstance(value, list):
        parts = [summarize_value(item, depth=depth + 1) for item in value[:max_items]]
        if len(value) > max_items:
            parts.append(f"... +{len(value) - max_items} more")
        return "[" + ", ".join(parts) + "]"
    return f"<{type(value).__name__}>"
