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
from cap_general.frameworks import import_frameworks


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
        return "base_scene"

    def __init__(self, config: BaseSceneConfig, logger: logging.Logger | None = None):
        self._config = config
        self._server_config = self._config.server
        self._record_dir = Path(self._config.record_dir).expanduser().resolve()
        self._logger = logger or self._build_logger(self._record_dir)
        self._agents: dict[str, BaseAgent] = {}
        self._agent_aliases: dict[str, str] = {}
        self._agent_status: dict[str, dict[str, Any]] = {}
        self._agent_tasks: dict[str, asyncio.Future[Any] | asyncio.Task[Any]] = {}
        set_current_scene(self)
        try:
            self._build_agents(self._config.agents)
        except BaseException:
            set_current_scene(None)
            raise

    @classmethod
    def from_yaml(cls, config_path: str | Path, overrides: list[str] | None = None) -> "BaseScene":
        """Initialize a scene from YAML with optional OmegaConf overrides."""
        return cls.from_config(load_yaml_config(config_path, overrides=overrides))

    @classmethod
    def get_server_url(cls, config_path: str | Path, overrides: list[str] | None = None) -> str:
        """Return the MCP server URL configured by a scene YAML file."""
        config_data = load_yaml_config(config_path, overrides=overrides)
        scene_type = config_data.pop("type")
        scene_cls = cls.get_registered_class(scene_type)
        if scene_cls is None:
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

    def _build_agents(self, specs: list[AgentSpec | dict[str, Any]]) -> None:
        self._before_build_agents()
        for spec_data in specs:
            spec = spec_data if isinstance(spec_data, AgentSpec) else AgentSpec(**spec_data)
            if spec.name in self._agents:
                raise ValueError(f"Duplicate agent name in scene: {spec.name}")
            agent_config = dict(spec.config)
            agent_config["record_dir"] = self._record_dir / spec.name
            aliases = spec.alias if isinstance(spec.alias, list) else [spec.alias] if spec.alias else []
            agent_config["name"] = spec.name
            agent_config["alias"] = aliases[0] if aliases else None
            agent = BaseAgent.from_config(agent_config, logger=self._logger)
            self._agents[spec.name] = agent
            for alias in aliases:
                existing = self._agent_aliases.get(alias)
                if existing is not None and existing != spec.name:
                    self._logger.warning(
                        "Skip duplicate agent alias %r for %s; already bound to %s", alias, spec.name, existing
                    )
                    continue
                self._agent_aliases[alias] = spec.name
        self._after_build_agents()

    def _before_build_agents(self) -> None:
        """Hook for subclasses to initialize state before agents."""

    def _after_build_agents(self) -> None:
        """Hook for subclasses to finalize state after agents."""

    def reset(self, agent_options: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Reset multiple agents from an agent-to-options mapping."""
        requests = self._resolve_kwargs(agent_options)
        results: dict[str, Any] = {}
        for canonical, options in requests.items():
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.reset(options=options)
        return results

    def agent_doc(self, agents: list[str]) -> dict[str, dict]:
        """Return documentation for the selected agents, or all agents if omitted."""
        results: dict[str, dict[str, Any]] = {}
        for canonical in self._resolve_names(agents):
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.agent_doc()
        return results

    async def execute(self, agent_codes: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Start code execution tasks for each selected agent."""
        requests = self._resolve_kwargs(agent_codes)
        results = [await self._start_task(agent, "execute", code=code) for agent, code in requests.items()]
        return self._format_results(requests, results)

    async def retry(self, agents: list[str]) -> dict[str, dict[str, Any]]:
        """Start retry tasks for selected agents."""
        canonical_agents = self._resolve_names(agents)
        results = [await self._start_task(agent, "retry") for agent in canonical_agents]
        return self._format_results(canonical_agents, results)

    async def train(self, agent_options: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Start train tasks for selected agents from an agent-to-options mapping."""
        requests = self._resolve_kwargs(agent_options)
        results = [await self._start_task(agent, "train", **kwargs) for agent, kwargs in requests.items()]
        return self._format_results(requests, results)

    async def monitor(self, agents: list[str], wait_ms: int = -1) -> dict[str, dict[str, Any]]:
        """Return selected agents' execution statuses, optionally waiting for completion."""
        canonical_agents = self._resolve_names(agents)
        if wait_ms != 0:
            await asyncio.gather(*(self._wait_task(agent, wait_ms) for agent in canonical_agents))
        return self._format_results(
            canonical_agents,
            [
                cap_utils.to_json_safe(self._agent_status.get(agent, self._get_status(agent)))
                for agent in canonical_agents
            ],
        )

    def record(self, agents: list[str]) -> dict[str, Any]:
        """Record complete run artifacts for selected agents, or all agents if omitted."""
        results: dict[str, Any] = {}
        for canonical in self._resolve_names(agents):
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.record(step_idx=-1)
        return results

    def update_history(self, agent_messages: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Append one history message to each selected agent transcript."""
        requests = self._resolve_kwargs(agent_messages)
        results: dict[str, Any] = {}
        for canonical, message in requests.items():
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.update_history(message)
        return results

    def get_obs(self, agents: list[str]) -> dict[str, Any]:
        """Return observations for the selected agents, or all agents if omitted."""
        results: dict[str, Any] = {}
        for canonical in self._resolve_names(agents):
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.get_obs()
        return results

    def set_trace_level(self, level: "cap_utils.TraceLevel | str") -> None:
        """Set the history trace level for all agents."""
        level = cap_utils.TraceLevel(level)
        for agent in self._agents.values():
            agent.set_trace_level(level)

    def serve(self, transport: str = "streamable-http") -> None:
        """Start an MCP server exposing scene-routed agent tools."""
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Serving a scene over MCP requires the mcp package") from exc

        s_config = self._server_config
        self.set_trace_level(cap_utils.TraceLevel.ALL)
        self._copy_skills_for_server(s_config)
        server = FastMCP(s_config.cap_id, host=s_config.host, port=s_config.port)
        for method_name in (
            "reset",
            "agent_doc",
            "execute",
            "train",
            "monitor",
            "retry",
            "record",
            "update_history",
            "get_obs",
        ):
            method = getattr(self, method_name)

            @functools.wraps(method)  # pylint: disable=cell-var-from-loop
            async def _wrapped(*args, _method=method, _name=method_name, **kwargs):
                self._logger.info(
                    "MCP scene tool call: %s args=%s kwargs=%s",
                    _name,
                    cap_utils.summarize_value(args),
                    cap_utils.summarize_value(kwargs),
                )
                result = _method(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            _wrapped.__doc__ = self._mcp_tool_doc(method_name, method)
            _wrapped.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
            server.tool()(_wrapped)

        self._logger.info(
            "Starting scene MCP server %s at http://%s:%s/mcp", s_config.cap_id, s_config.host, s_config.port
        )
        import anyio

        anyio.run(self._run_server_async, server, transport)

    async def _start_task(self, canonical: str, method_name: str, **kwargs: Any) -> dict[str, Any]:
        task = self._agent_tasks.get(canonical)
        if task is not None and not task.done():
            return cap_utils.to_json_safe(self._agent_status.get(canonical, self._get_status(canonical)))
        started_at = time.time()
        self._agent_status[canonical] = self._get_status(
            canonical, method=method_name, running=True, started_at=started_at
        )
        agent = self._get_agent(canonical)
        method = getattr(agent, method_name)

        async def _run() -> dict[str, Any]:
            try:
                result = await self._dispatch_task(method, kwargs)
            except BaseException as exc:
                self._logger.exception("Agent task failed: %s.%s", canonical, method_name)
                result = {"ok": False, "error": type(exc).__name__, "err_msg": str(exc)}
            finished_at = time.time()
            status = self._get_status(
                canonical,
                method=method_name,
                started_at=started_at,
                finished_at=finished_at,
                duration=f"{finished_at - started_at:.2f}s",
                result=cap_utils.to_json_safe(result),
            )
            self._agent_status[canonical] = status
            return status

        self._agent_tasks[canonical] = asyncio.create_task(_run())
        return cap_utils.to_json_safe(self._agent_status.get(canonical, self._get_status(canonical)))

    async def _dispatch_task(self, method: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Dispatch a synchronous agent method without blocking the event loop."""
        return await asyncio.to_thread(method, **kwargs)

    async def _wait_task(self, canonical: str, wait_ms: int) -> None:
        task = self._agent_tasks.get(canonical)
        if task is None or task.done():
            return
        if wait_ms < 0:
            await asyncio.shield(task)
            return
        if wait_ms == 0:
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=max(wait_ms, 0) / 1000.0)
        except asyncio.TimeoutError:
            return

    def _get_status(self, canonical: str, **kwargs: Any) -> dict[str, Any]:
        status = {
            "agent": canonical,
            "method": None,
            "running": False,
            "started_at": None,
            "finished_at": None,
            "duration": None,
            "result": None,
        }
        status.update(kwargs)
        return status

    async def _run_server_async(self, server: Any, transport: str) -> None:
        """Run the MCP server inside the event loop.

        Separated from ``serve()`` so subclasses can override
        ``_on_server_started()`` to schedule async startup work (e.g. idle
        render loops) after the event loop is running but before the server
        begins accepting requests.
        """
        await self._on_server_started()
        if transport == "streamable-http":
            await server.run_streamable_http_async()
        elif transport == "sse":
            await server.run_sse_async()
        elif transport == "stdio":
            await server.run_stdio_async()
        else:
            raise ValueError(f"Unknown transport: {transport!r}")

    async def _on_server_started(self) -> None:
        """Called once the event loop is running, before the server accepts requests.

        Override in subclasses to schedule asyncio tasks that require a live
        event loop (e.g. background render loops).
        """

    def _get_agent(self, agent: str | None = None) -> BaseAgent:
        """Return an agent by name or alias."""
        return self._agents[self._resolve_name(agent)]

    def _resolve_name(self, agent: str | None = None) -> str:
        """Return the canonical agent name for a name or alias."""
        if agent is None:
            if len(self._agents) == 1:
                return next(iter(self._agents))
            raise ValueError("agent is required when a scene has multiple agents")
        canonical = self._agent_aliases.get(agent, agent)
        if canonical not in self._agents:
            raise KeyError(f"Unknown agent: {agent}")
        return canonical

    def _resolve_names(self, agents: list[str] | None = None) -> list[str]:
        """Resolve names and aliases to unique canonical agent names."""
        requested = list(self._agents) if agents is None else agents
        canonical_agents: list[str] = []
        for agent in requested:
            canonical = self._resolve_name(agent)
            if canonical not in canonical_agents:
                canonical_agents.append(canonical)
        return canonical_agents

    def _resolve_kwargs(self, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve mapping keys to canonical names and reject alias collisions."""
        resolved: dict[str, Any] = {}
        for agent, value in values.items():
            canonical = self._resolve_name(agent)
            if canonical in resolved:
                raise ValueError(f"Duplicate agent mapping after alias resolution: {agent!r} -> {canonical!r}")
            resolved[canonical] = value
        return resolved

    def _format_results(self, agents: Iterable[str], results: Iterable[Any]) -> dict[str, Any]:
        """Key ordered agent results by their human-readable response names."""
        formatted: dict[str, Any] = {}
        for canonical, result in zip(agents, results, strict=True):
            agent = self._get_agent(canonical)
            formatted[agent.mark] = result
        return formatted

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
