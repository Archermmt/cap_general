"""Shared Genesis scene construction helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SceneConfig:
    """Configuration for the process-global Genesis scene."""

    show_viewer: bool = False
    sim_options: dict[str, Any] = field(default_factory=dict)
    rigid_options: dict[str, Any] = field(default_factory=dict)
    viewer_options: dict[str, Any] = field(default_factory=dict)
    vis_options: dict[str, Any] = field(default_factory=dict)
    profiling_options: dict[str, Any] = field(default_factory=dict)


_SCENE: Any | None = None
_SCENE_CONFIG: SceneConfig | None = None


def get_scene(config: SceneConfig | dict[str, Any] | None = None, *, gs: Any | None = None) -> Any:
    """Create or return the process-global Genesis scene."""
    global _SCENE, _SCENE_CONFIG

    if _SCENE is not None:
        return _SCENE
    if gs is None:
        import genesis as gs

    scene_config = coerce_scene_config(config)
    kwargs: dict[str, Any] = {"show_viewer": scene_config.show_viewer}
    if scene_config.sim_options:
        kwargs["sim_options"] = gs.options.SimOptions(**_resolve_options(scene_config.sim_options, gs))
    if scene_config.rigid_options:
        kwargs["rigid_options"] = gs.options.RigidOptions(**_resolve_options(scene_config.rigid_options, gs))
    if scene_config.viewer_options:
        kwargs["viewer_options"] = gs.options.ViewerOptions(**_resolve_options(scene_config.viewer_options, gs))
    if scene_config.vis_options:
        kwargs["vis_options"] = gs.options.VisOptions(**_resolve_options(scene_config.vis_options, gs))
    if scene_config.profiling_options:
        kwargs["profiling_options"] = gs.options.ProfilingOptions(**_resolve_options(scene_config.profiling_options, gs))
    _SCENE = gs.Scene(**kwargs)
    _SCENE_CONFIG = scene_config
    return _SCENE


def coerce_scene_config(config: SceneConfig | dict[str, Any] | None) -> SceneConfig:
    """Return a SceneConfig from a dataclass, dict, or None."""
    if config is None:
        return SceneConfig()
    if isinstance(config, SceneConfig):
        return config
    if isinstance(config, dict):
        return SceneConfig(**config)
    raise TypeError(f"Expected SceneConfig or dict, got {type(config).__name__}")


def reset_scene() -> None:
    """Forget the process-global scene reference."""
    global _SCENE, _SCENE_CONFIG
    _SCENE = None
    _SCENE_CONFIG = None


def _resolve_options(options: dict[str, Any], gs: Any) -> dict[str, Any]:
    resolved = dict(options)
    solver = resolved.get("constraint_solver")
    if isinstance(solver, str):
        resolved["constraint_solver"] = getattr(gs.constraint_solver, solver)
    return resolved
