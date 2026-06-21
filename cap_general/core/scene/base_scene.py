"""Top-level scene that owns agents and the MCP server."""

from __future__ import annotations

import functools
import inspect
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core import utils as cap_utils
from cap_general.core.agent import BaseAgent
from cap_general.core.base import RegisteredBase
from cap_general.core.scene.context import set_current_scene


@dataclass
class ServerConfig:
    """Configuration for the MCP server used by a scene."""

    cap_id: str = "cap"
    host: str = "127.0.0.1"
    port: int = 8080
    skill_folder: str = "skills"


@dataclass
class AgentSpec:
    """Configuration for an agent inside a scene."""

    name: str
    config: dict[str, Any]
    alias: str | list[str] | None = None


@dataclass
class BaseSceneConfig:
    """Configuration for a scene containing one or more agents."""

    agents: list[AgentSpec | dict[str, Any]]
    server: ServerConfig = field(default_factory=ServerConfig)
    record_dir: str | Path = "outputs/scene"


class BaseScene(RegisteredBase):
    """A scene contains multiple named agents."""

    name = "Base Scene"
    _registry: ClassVar[dict[str, type["BaseScene"]]] = {}
    config_cls: ClassVar[type[BaseSceneConfig]] = BaseSceneConfig
    registry_key_method: ClassVar[str] = "scene_type"

    @classmethod
    def scene_type(cls) -> str:
        """Return the registry key for this scene."""
        return "scene"

    def __init__(self, config: BaseSceneConfig, logger: logging.Logger | None = None):
        self._config = config
        self._server_config = self._config.server
        self._record_dir = Path(self._config.record_dir).expanduser().resolve()
        self._logger = logger or self._build_logger(self._record_dir)
        self._agents: dict[str, BaseAgent] = {}
        self._agent_aliases: dict[str, str] = {}
        self._before_build_agents()
        set_current_scene(self)
        try:
            self._build_agents(self._config.agents)
        finally:
            set_current_scene(None)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BaseScene":
        """Initialize a scene from a YAML config file."""
        return cls.from_config(cls._load_yaml_config(config_path))

    @staticmethod
    def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
        """Load a scene YAML config."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load scene yaml configs") from exc

        path = Path(config_path)
        with path.open() as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise TypeError("Scene yaml config must contain a mapping at the top level")
        data.setdefault("type", "scene")
        return data

    @classmethod
    def get_server_url(cls, config_path: str | Path) -> str:
        """Return the MCP server URL configured by a scene YAML file."""
        config_data = cls._load_yaml_config(config_path)
        scene_type = config_data.pop("type")
        scene_cls = cls.get_registered_class(scene_type)
        if scene_cls is None:
            from cap_general.frameworks import import_frameworks

            import_frameworks()
            scene_cls = cls.get_registered_class(scene_type)
        if scene_cls is None:
            raise KeyError(f"Unknown registered type: {scene_type}")
        config_obj = cls._build_dataclass_config(scene_cls.config_cls, config_data)
        s_config = config_obj.server
        return f"http://{s_config.host}:{s_config.port}/mcp"

    @staticmethod
    def _build_logger(record_dir: Path) -> logging.Logger:
        return cap_utils.build_file_logger(record_dir, logger_name="scene")

    def _before_build_agents(self) -> None:
        """Hook for subclasses to initialize state before agents."""

    def _build_agents(self, specs: list[AgentSpec | dict[str, Any]]) -> None:
        for spec_data in specs:
            spec = spec_data if isinstance(spec_data, AgentSpec) else AgentSpec(**spec_data)
            if spec.name in self._agents:
                raise ValueError(f"Duplicate agent name in scene: {spec.name}")
            agent_config = dict(spec.config)
            agent_config["record_dir"] = self._record_dir / spec.name
            agent = BaseAgent.from_config(agent_config, logger=self._logger)
            self._agents[spec.name] = agent
            self._agent_aliases[spec.name] = spec.name
            aliases = spec.alias if isinstance(spec.alias, list) else [spec.alias] if spec.alias else []
            for alias in aliases:
                existing = self._agent_aliases.get(alias)
                if existing is not None and existing != spec.name:
                    self._logger.warning(
                        "Skip duplicate agent alias %r for %s; already bound to %s", alias, spec.name, existing
                    )
                    continue
                self._agent_aliases[alias] = spec.name

    def reset(self, agent: str | None = None, options: dict[str, Any] | None = None):
        """Reset one agent by name or alias."""
        return self._get_agent(agent).reset(options=options)

    def agent_doc(self, agent: str | None = None) -> dict:
        """Return one agent's documentation."""
        return self._get_agent(agent).agent_doc()

    def execute(self, agent: str | None = None, code: str = ""):
        """Execute code on one agent by name or alias."""
        return self._get_agent(agent).execute(code)

    def retry(self, agent: str | None = None):
        """Retry the latest execution on one agent."""
        return self._get_agent(agent).retry()

    def record(self, agent: str | None = None, step_idx: int = -1):
        """Record artifacts for one agent."""
        return self._get_agent(agent).record(step_idx=step_idx)

    def update_plan(self, agent: str | None = None, plan: dict[str, Any] | None = None):
        """Update one agent's plan."""
        return self._get_agent(agent).update_plan(plan or {})

    def get_obs(self, agent: str | None = None):
        """Return one agent's current observation."""
        return self._get_agent(agent).get_obs()

    def serve(self, transport: str = "streamable-http") -> None:
        """Start an MCP server exposing scene-routed agent tools."""
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Serving a scene over MCP requires the mcp package") from exc

        s_config = self._server_config
        self._copy_skills_for_server(s_config)
        server = FastMCP(s_config.cap_id, host=s_config.host, port=s_config.port)
        for method_name in ("reset", "agent_doc", "execute", "retry", "record", "update_plan", "get_obs"):
            method = getattr(self, method_name)

            @functools.wraps(method)  # pylint: disable=cell-var-from-loop
            def _wrapped(*args, _method=method, _name=method_name, **kwargs):
                self._logger.info(
                    "MCP scene tool call: %s args=%s kwargs=%s",
                    _name,
                    cap_utils.summarize_value(args),
                    cap_utils.summarize_value(kwargs),
                )
                return _method(*args, **kwargs)

            _wrapped.__doc__ = self._mcp_tool_doc(method_name, method)
            _wrapped.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
            server.tool()(_wrapped)

        self._logger.info(
            "Starting scene MCP server %s at http://%s:%s/mcp", s_config.cap_id, s_config.host, s_config.port
        )
        server.run(transport=transport)

    def _get_agent(self, agent: str | None = None) -> BaseAgent:
        """Return an agent by name or alias."""
        if agent is None:
            if len(self._agents) == 1:
                return next(iter(self._agents.values()))
            raise ValueError("agent is required when a scene has multiple agents")
        canonical = self._agent_aliases.get(agent)
        if canonical is None:
            raise KeyError(f"Unknown agent: {agent}")
        return self._agents[canonical]

    def _copy_skills_for_server(self, server_config: ServerConfig) -> Path:
        """Render scene-bound skills under ``skill_folder/cap_id``."""
        source_dir = Path(__file__).resolve().parents[2] / "skills"
        skill_root = Path(server_config.skill_folder).expanduser()
        target_dir = skill_root / server_config.cap_id
        if not skill_root.exists() or not any(skill_root.iterdir()):
            self._logger.info("Skip copying skills because skill_folder is missing or empty: %s", skill_root)
            return target_dir
        if not source_dir.exists():
            self._logger.warning("Skill source directory does not exist: %s", source_dir)
            return target_dir
        if target_dir.resolve() == source_dir.resolve():
            raise ValueError("server.skill_folder/cap_id must not point to the source skills directory")

        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        replacements = {
            "{cap_id}": server_config.cap_id,
            "{available_names}": ", ".join(sorted(self._agent_aliases)),
        }
        for source_path in source_dir.rglob("*"):
            if "__pycache__" in source_path.parts:
                continue
            target_path = target_dir / source_path.relative_to(source_dir)
            if source_path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.name == "SKILL.md":
                content = source_path.read_text(encoding="utf-8")
                for old, new in replacements.items():
                    content = content.replace(old, new)
                target_path.write_text(content, encoding="utf-8")
            else:
                shutil.copy2(source_path, target_path)
        return target_dir

    @staticmethod
    def _mcp_tool_doc(method_name: str, method: Callable[..., Any]) -> str:
        doc = inspect.getdoc(method) or ""
        return f"{doc}\n\nArgs:\n    agent: Agent name or alias to route this call to."

    @property
    def server_config(self) -> ServerConfig:
        """Return this scene's MCP server configuration."""
        return self._server_config

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Return agents keyed by canonical name."""
        return dict(self._agents)


BaseScene._registry[BaseScene.scene_type()] = BaseScene
