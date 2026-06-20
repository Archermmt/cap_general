"""Process-local active scene context."""

from __future__ import annotations

from typing import Any

_CURRENT_SCENE: Any | None = None


def set_current_scene(scene: Any | None) -> None:
    """Set the process-local scene used while constructing envs."""
    global _CURRENT_SCENE
    _CURRENT_SCENE = scene


def get_current_scene() -> Any | None:
    """Return the process-local scene, if one has been installed."""
    return _CURRENT_SCENE
