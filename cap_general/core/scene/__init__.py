"""Core CAP scene components."""

from cap_general.core.scene.context import get_current_scene, set_current_scene

__all__ = [
    "AgentSpec",
    "BaseScene",
    "BaseSceneConfig",
    "ServerConfig",
    "get_current_scene",
    "set_current_scene",
]


def __getattr__(name):
    """Lazily expose scene classes to avoid core import cycles."""
    if name in {"AgentSpec", "BaseScene", "BaseSceneConfig", "ServerConfig"}:
        from cap_general.core.scene.base_scene import AgentSpec, BaseScene, BaseSceneConfig, ServerConfig

        values = {
            "AgentSpec": AgentSpec,
            "BaseScene": BaseScene,
            "BaseSceneConfig": BaseSceneConfig,
            "ServerConfig": ServerConfig,
        }
        return values[name]
    raise AttributeError(name)
