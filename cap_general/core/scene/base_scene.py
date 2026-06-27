"""Top-level scene that owns agents and the MCP server."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core import utils as cap_utils
from cap_general.core.agent import BaseAgent
from cap_general.core.base import RegisteredBase
from cap_general.core.scene.context import set_current_scene
from cap_general.core.utils.config import load_yaml_config


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
        self._agent_execute_tasks: dict[str, Any] = {}
        self._agent_execute_status: dict[str, dict[str, Any]] = {}
        self._background_event_task: Any | None = None
        self._before_build_agents()
        set_current_scene(self)
        try:
            self._build_agents(self._config.agents)
            self._after_build_agents()
        finally:
            set_current_scene(None)

    @classmethod
    def from_yaml(cls, config_path: str | Path, overrides: list[str] | None = None) -> "BaseScene":
        """Initialize a scene from YAML with optional OmegaConf overrides."""
        return cls.from_config(cls._load_yaml_config(config_path, overrides=overrides))

    @staticmethod
    def _load_yaml_config(
        config_path: str | Path, overrides: list[str] | None = None
    ) -> dict[str, Any]:
        """Load a scene YAML config and apply recursive overrides."""
        return load_yaml_config(config_path, overrides=overrides)

    @classmethod
    def get_server_url(cls, config_path: str | Path, overrides: list[str] | None = None) -> str:
        """Return the MCP server URL configured by a scene YAML file."""
        config_data = cls._load_yaml_config(config_path, overrides=overrides)
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

    def _after_build_agents(self) -> None:
        """Hook for subclasses to finalize state after agents."""

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

    def reset(self, agent_options: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Reset multiple agents from an agent-to-options mapping."""
        requests = self._resolve_agent_mapping(agent_options)
        return {
            self._agent_mark(canonical): self._agents[canonical].reset(options=options)
            for canonical, options in requests.items()
        }

    def agent_doc(self, agents: list[str]) -> dict[str, dict]:
        """Return documentation for the selected agents, or all agents if omitted."""
        return {
            self._agent_mark(canonical): self._agents[canonical].agent_doc()
            for canonical in self._resolve_agent_names(agents)
        }

    async def execute(self, agent_codes: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Start code execution concurrently from an agent-to-code mapping.

        An agent with an unfinished execution keeps running its current task and
        does not start the newly supplied code.
        """
        requests = self._resolve_agent_mapping(agent_codes)
        statuses = await asyncio.gather(
            *(self._start_agent_task(agent, "execute", code) for agent, code in requests.items())
        )
        return self._format_agent_results(requests, statuses)

    async def retry(self, agents: list[str]) -> dict[str, dict[str, Any]]:
        """Start retrying the latest execution concurrently for selected agents."""
        canonical_agents = self._resolve_agent_names(agents)
        statuses = await asyncio.gather(*(self._start_agent_task(agent, "retry") for agent in canonical_agents))
        return self._format_agent_results(canonical_agents, statuses)

    async def _start_agent_task(self, agent: str, method_name: str, *args: Any) -> dict[str, Any]:
        canonical = self._resolve_agent_name(agent)
        task = self._agent_execute_tasks.get(canonical)
        if task is not None and not task.done():
            return await self._monitor_agent(canonical, wait_ms=0)

        started_at = time.time()
        self._agent_execute_status[canonical] = {
            "agent": canonical,
            "method": method_name,
            "running": True,
            "started_at": started_at,
            "finished_at": None,
            "duration": None,
            "result": None,
            "error": None,
        }

        task = asyncio.create_task(self._run_agent_task(canonical, method_name, started_at, *args))
        self._agent_execute_tasks[canonical] = task
        self._ensure_background_event_pump()
        return await self._monitor_agent(canonical, wait_ms=0)

    async def monitor(self, agents: list[str], wait_ms: int = -1) -> dict[str, dict[str, Any]]:
        """Return selected agents' statuses after applying a shared wait policy.

        Args:
            agents: Agent names or aliases to inspect.
            wait_ms: ``-1`` waits for completion, ``0`` returns immediately,
                and a positive value waits that many milliseconds before
                returning the latest status.
        """
        if wait_ms < -1:
            raise ValueError("wait_ms must be -1, 0, or a positive integer")
        canonical_agents = self._resolve_agent_names(agents)
        statuses = await asyncio.gather(
            *(self._monitor_agent(agent, wait_ms) for agent in canonical_agents)
        )
        return self._format_agent_results(canonical_agents, statuses)

    async def _monitor_agent(self, agent: str, wait_ms: int) -> dict[str, Any]:
        """Return one canonical agent's current execution status.

        Args:
            agent: Canonical agent name to inspect.
            wait_ms: Wait policy in milliseconds; see :meth:`monitor`.
        """
        task = self._agent_execute_tasks.get(agent)
        status = self._agent_execute_status.get(
            agent,
            {
                "agent": agent,
                "method": None,
                "running": False,
                "started_at": None,
                "finished_at": None,
                "duration": None,
                "result": None,
                "error": None,
            },
        )
        if task is None:
            if wait_ms > 0:
                await asyncio.sleep(wait_ms / 1000)
            return cap_utils.to_json_safe(status)
        if wait_ms == -1 and not task.done():
            self._ensure_background_event_pump()
            status = await task
        elif wait_ms > 0:
            if not task.done():
                self._ensure_background_event_pump()
            await asyncio.sleep(wait_ms / 1000)
            status = self._agent_execute_status.get(agent, status)
        elif task.done():
            status = self._agent_execute_status.get(agent, status)
        self._process_background_events()
        return cap_utils.to_json_safe(status)

    def _process_background_events(self) -> bool:
        """Process subclass-owned main-thread events while agent tasks run."""
        return False

    def _has_running_agent_tasks(self) -> bool:
        return any(task is not None and not task.done() for task in self._agent_execute_tasks.values())

    def _ensure_background_event_pump(self) -> None:
        if self._background_event_task is None or self._background_event_task.done():
            self._background_event_task = asyncio.create_task(self._background_event_pump())

    async def _background_event_pump(self) -> None:
        try:
            while self._has_running_agent_tasks():
                processed = self._process_background_events()
                await asyncio.sleep(0 if processed else 0.001)
            self._process_background_events()
        finally:
            self._background_event_task = None

    def record(self, agents: list[str]) -> dict[str, Any]:
        """Record complete run artifacts for selected agents, or all agents if omitted."""
        return {
            self._agent_mark(canonical): self._agents[canonical].record(step_idx=-1)
            for canonical in self._resolve_agent_names(agents)
        }

    def update_plan(self, agent_plans: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Update plans from an agent-to-plan mapping."""
        requests = self._resolve_agent_mapping(agent_plans)
        return {
            self._agent_mark(canonical): self._agents[canonical].update_plan(plan)
            for canonical, plan in requests.items()
        }

    def get_obs(self, agents: list[str]) -> dict[str, Any]:
        """Return observations for the selected agents, or all agents if omitted."""
        return {
            self._agent_mark(canonical): self._agents[canonical].get_obs()
            for canonical in self._resolve_agent_names(agents)
        }

    def serve(self, transport: str = "streamable-http") -> None:
        """Start an MCP server exposing scene-routed agent tools."""
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Serving a scene over MCP requires the mcp package") from exc

        s_config = self._server_config
        self._copy_skills_for_server(s_config)
        server = FastMCP(s_config.cap_id, host=s_config.host, port=s_config.port)
        for method_name in ("reset", "agent_doc", "execute", "monitor", "retry", "record", "update_plan", "get_obs"):
            method = getattr(self, method_name)

            if inspect.iscoroutinefunction(method):

                @functools.wraps(method)  # pylint: disable=cell-var-from-loop
                async def _wrapped(*args, _method=method, _name=method_name, **kwargs):
                    self._logger.info(
                        "MCP scene tool call: %s args=%s kwargs=%s",
                        _name,
                        cap_utils.summarize_value(args),
                        cap_utils.summarize_value(kwargs),
                    )
                    return await _method(*args, **kwargs)

            else:

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

    async def _run_agent_task(self, canonical: str, method_name: str, started_at: float, *args: Any) -> dict[str, Any]:
        try:
            method = getattr(self._agents[canonical], method_name)
            result = await asyncio.to_thread(method, *args)
            error = None
        except BaseException as exc:  # pragma: no cover - defensive task wrapper
            result = {
                "ok": False,
                "error": type(exc).__name__,
                "stderr": str(exc),
            }
            error = {"type": type(exc).__name__, "message": str(exc)}

        finished_at = time.time()
        status = {
            "agent": canonical,
            "method": method_name,
            "running": False,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration": f"{finished_at - started_at:.2f}s",
            "result": cap_utils.to_json_safe(result),
            "error": error,
        }
        self._agent_execute_status[canonical] = status
        return status

    def _get_agent(self, agent: str | None = None) -> BaseAgent:
        """Return an agent by name or alias."""
        return self._agents[self._resolve_agent_name(agent)]

    def _resolve_agent_name(self, agent: str | None = None) -> str:
        """Return the canonical agent name for a name or alias."""
        if agent is None:
            if len(self._agents) == 1:
                return next(iter(self._agents))
            raise ValueError("agent is required when a scene has multiple agents")
        canonical = self._agent_aliases.get(agent)
        if canonical is None:
            raise KeyError(f"Unknown agent: {agent}")
        return canonical

    def _resolve_agent_names(self, agents: list[str] | None = None) -> list[str]:
        """Resolve names and aliases to unique canonical agent names."""
        requested = list(self._agents) if agents is None else agents
        canonical_agents: list[str] = []
        for agent in requested:
            canonical = self._resolve_agent_name(agent)
            if canonical not in canonical_agents:
                canonical_agents.append(canonical)
        return canonical_agents

    def _resolve_agent_mapping(self, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve mapping keys to canonical names and reject alias collisions."""
        resolved: dict[str, Any] = {}
        for agent, value in values.items():
            canonical = self._resolve_agent_name(agent)
            if canonical in resolved:
                raise ValueError(f"Duplicate agent mapping after alias resolution: {agent!r} -> {canonical!r}")
            resolved[canonical] = value
        return resolved

    def _agent_mark(self, canonical: str) -> str:
        """Return a human-readable agent mark such as ``alias(agent_name)``."""
        alias = next(
            (name for name, agent_name in self._agent_aliases.items() if agent_name == canonical and name != canonical),
            None,
        )
        return f"{alias}({canonical})" if alias else canonical

    def _format_agent_results(self, agents: Iterable[str], results: Iterable[Any]) -> dict[str, Any]:
        """Key ordered agent results by their human-readable response names."""
        return {self._agent_mark(canonical): result for canonical, result in zip(agents, results, strict=True)}

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
        return inspect.getdoc(method) or method_name

    @property
    def server_config(self) -> ServerConfig:
        """Return this scene's MCP server configuration."""
        return self._server_config

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Return agents keyed by canonical name."""
        return dict(self._agents)


BaseScene._registry[BaseScene.scene_type()] = BaseScene
