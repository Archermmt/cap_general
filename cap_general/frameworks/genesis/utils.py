"""Genesis runtime utilities."""

from __future__ import annotations

def step_scene(scene) -> None:
    """Step a Genesis scene directly."""
    bridge = getattr(scene, "_cap_step_scene", None)
    if callable(bridge):
        bridge()
        return
    scene.step()
