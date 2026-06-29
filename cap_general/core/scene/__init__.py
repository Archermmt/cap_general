"""Core CAP scene components."""

__all__ = [
    "AgentSpec",
    "BaseScene",
    "BaseSceneConfig",
    "ServerConfig",
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
