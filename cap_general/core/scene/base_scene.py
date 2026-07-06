"""Top-level scene that owns agents and the MCP server."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core import utils as cap_utils
from cap_general.core.agent import BaseAgent
from cap_general.core.base import RegisteredBase
from cap_general.core.utils.config import build_dataclass_config, load_yaml_config
from cap_general.frameworks import import_frameworks


@dataclass
class ServerConfig:
    """Configuration for the MCP server used by a scene."""

    cap_id: str = "cap"
    host: str = "127.0.0.1"
    port: int = 8080
    skill_folder: str = "skills"
    fast: bool = True


@dataclass
class AgentInfo:
    """Runtime state for one scene agent."""

    agent: BaseAgent
    status: dict[str, Any]
    task: asyncio.Future[Any] | asyncio.Task[Any] | None = None


@dataclass
class BaseSceneConfig:
    """Configuration for a scene containing one or more agents."""

    agents: list[dict[str, Any]]
    server: ServerConfig = field(default_factory=ServerConfig)
    record_dir: str | Path = "outputs/scene"
    export_dir: str | Path = "outputs/export"
    debug: bool = False
    async_task: bool = False
    trace_level: cap_utils.TraceLevel | str = cap_utils.TraceLevel.ALL


@RegisteredBase.register()
class BaseScene(RegisteredBase):
    """A scene contains multiple named agents."""

    _registry: ClassVar[dict[str, type["BaseScene"]]] = {}
    registry_key_attr: ClassVar[str] = "scene_type"
    scene_type: ClassVar[str] = "base"
    config_cls: ClassVar[type[BaseSceneConfig]] = BaseSceneConfig

    @staticmethod
    def trace_result(method: Callable[..., Any]) -> Callable[..., Any]:
        """Trace one request and response per agent routed by a scene method."""
        signature = inspect.signature(method)

        def should_trace(self: "BaseScene") -> bool:
            task_methods = {"execute", "retry", "monitor", "get_obs"}
            return self._trace_level is cap_utils.TraceLevel.ALL or (
                self._trace_level is cap_utils.TraceLevel.TASK and method.__name__ in task_methods
            )

        def trace_request(self: "BaseScene", args: tuple[Any, ...], kwargs: dict[str, Any]):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            arguments.pop("self", None)
            self._append_history(
                {"role": "user", "tool": method.__name__, "request": cap_utils.to_json_safe(arguments)}
            )

        def trace_response(self: "BaseScene", results: dict[str, Any]) -> None:
            for agent_mark, result in results.items():
                redundant_keys = {"agent"}
                if method.__name__ != "monitor":
                    redundant_keys.add("method")
                response = (
                    {key: value for key, value in result.items() if key not in redundant_keys}
                    if isinstance(result, dict)
                    else result
                )
                self._append_history(
                    {
                        "role": agent_mark,
                        "tool": method.__name__,
                        "response": cap_utils.to_json_safe(response),
                    }
                )

        if inspect.iscoroutinefunction(method):

            @functools.wraps(method)
            async def async_wrapper(self: "BaseScene", *args: Any, **kwargs: Any) -> Any:
                tracing = should_trace(self)
                if tracing:
                    trace_request(self, args, kwargs)
                result = await method(self, *args, **kwargs)
                if tracing:
                    trace_response(self, result)
                return result

            return async_wrapper

        @functools.wraps(method)
        def wrapper(self: "BaseScene", *args: Any, **kwargs: Any) -> Any:
            tracing = should_trace(self)
            if tracing:
                trace_request(self, args, kwargs)
            result = method(self, *args, **kwargs)
            if tracing:
                trace_response(self, result)
            return result

        return wrapper

    def __init__(self, config: BaseSceneConfig, logger: logging.Logger | None = None):
        self._config = config
        self._server_config = self._config.server
        self._record_dir = Path(self._config.record_dir).expanduser().resolve()
        self._export_dir = Path(self._config.export_dir).expanduser().resolve()
        self._logger = logger or self._build_logger(self._record_dir)
        self._trace_level = cap_utils.TraceLevel(self._config.trace_level)
        self._history: list[dict[str, Any]] = []
        cap_utils.remove_path(self._record_dir / "history.json")
        self._agent_aliases: dict[str, str] = {}
        self._agents: dict[str, AgentInfo] = {}
        self._build_agents(self._config.agents)

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
        config_obj = build_dataclass_config(scene_cls.config_cls, config_data)
        s_config = config_obj.server
        return f"http://{s_config.host}:{s_config.port}/mcp"

    @staticmethod
    def _build_logger(record_dir: Path) -> logging.Logger:
        return cap_utils.build_file_logger(record_dir, logger_name="scene")

    def _build_agents(self, specs: list[dict[str, Any]]) -> None:
        self._pre_build()
        for spec_data in specs:
            raw = dict(spec_data)
            name = raw.pop("name")
            alias = raw.pop("alias", None)
            if name in self._agents:
                raise ValueError(f"Duplicate agent name in scene: {name}")
            raw.pop("debug", None)
            raw["record_dir"] = self._record_dir / name
            raw["debug"] = self._config.debug
            raw["name"] = name
            raw["alias"] = alias
            if raw.get("pipeline"):
                pipeline = dict(raw["pipeline"])
                pipeline["export_dir"] = str(self._export_dir / name)
                raw["pipeline"] = pipeline
            agent = BaseAgent.from_config(raw, logger=self._logger)
            self._agents[name] = AgentInfo(agent=agent, status=self._get_status(name))
            if alias:
                existing = self._agent_aliases.get(alias)
                if existing is not None and existing != name:
                    self._logger.warning(
                        "Skip duplicate agent alias %r for %s; already bound to %s", alias, name, existing
                    )
                else:
                    self._agent_aliases[alias] = name
        self._post_build()

    def _pre_build(self) -> None:
        """Hook called before agents are constructed."""

    def _post_build(self) -> None:
        """Hook called after all agents are constructed."""
        for agent_info in self._agents.values():
            agent_info.agent.post_build(self)

    def reset(self, agent_options: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Reset multiple agents from an agent-to-options mapping."""
        requests = self._resolve_kwargs(agent_options)
        results: dict[str, Any] = {}
        for canonical, options in requests.items():
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.reset(options=options)
        return results

    @trace_result
    def agent_doc(self, agents: list[str]) -> dict[str, Any]:
        """Return documentation for the selected agents, or all agents if omitted."""
        agents_doc = {}
        for canonical in self._resolve_names(agents):
            agent = self._get_agent(canonical)
            agents_doc[agent.mark] = agent.agent_doc()
        return {
            "scene": {
                "async_task": self._config.async_task,
                "agents": agents_doc,
            }
        }

    @trace_result
    async def execute(self, agent_codes: dict[str, str]) -> dict[str, dict[str, Any]]:
        """Start code execution tasks for each selected agent."""
        requests = self._resolve_kwargs(agent_codes)
        results = [await self._start_task(agent, "execute", code=code) for agent, code in requests.items()]
        return self._format_results(requests, results)

    @trace_result
    async def retry(self, agents: list[str]) -> dict[str, dict[str, Any]]:
        """Start retry tasks for selected agents."""
        canonical_agents = self._resolve_names(agents)
        results = [await self._start_task(agent, "retry") for agent in canonical_agents]
        return self._format_results(canonical_agents, results)

    @trace_result
    async def run_pipe(self, agent_options: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Run pipeline jobs for selected agents from an agent-to-options mapping."""
        requests = self._resolve_kwargs(agent_options)
        results = [await self._start_task(agent, "run_pipe", **kwargs) for agent, kwargs in requests.items()]
        return self._format_results(requests, results)

    @trace_result
    async def monitor(self, agents: list[str], wait_ms: int = -1) -> dict[str, dict[str, Any]]:
        """Return selected agents' execution statuses, optionally waiting for completion."""
        canonical_agents = self._resolve_names(agents)
        if wait_ms != 0:
            await asyncio.gather(*(self._wait_task(agent, wait_ms) for agent in canonical_agents))
        return self._format_results(
            canonical_agents,
            [cap_utils.to_json_safe(self._agents[agent].status) for agent in canonical_agents],
        )

    @trace_result
    def get_obs(self, agents: list[str]) -> dict[str, Any]:
        """Return observations for the selected agents, or all agents if omitted."""
        results: dict[str, Any] = {}
        for canonical in self._resolve_names(agents):
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.get_obs()
        return results

    def record(self, agents: list[str], clean_frames: bool = False) -> dict[str, Any]:
        """Record complete run artifacts for selected agents, or all agents if omitted."""
        results: dict[str, Any] = {}
        for canonical in self._resolve_names(agents):
            agent = self._get_agent(canonical)
            results[agent.mark] = agent.record(step_idx=-1, clean_frames=clean_frames)
        return results

    def update_history(self, agent_messages: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Append one history message per selected agent to the scene transcript."""
        requests = self._resolve_kwargs(agent_messages)
        results: dict[str, Any] = {}
        for canonical, message in requests.items():
            agent = self._get_agent(canonical)
            results[agent.mark] = self._append_history(message, canonical=canonical)
        return results

    def _append_history(self, message: dict[str, Any], canonical: str | None = None) -> dict[str, Any]:
        """Persist one message in the scene history."""
        if not isinstance(message, dict):
            raise TypeError("message must be a history message dictionary")
        hist_message = {
            "timestamp": datetime.now().strftime("%Y-%m-%d:%H-%M-%S.%f")[:-3],
            **message,
        }
        if canonical is not None:
            hist_message["agent"] = self._get_agent(canonical).mark
        self._history.append(hist_message)
        history_path = self._record_dir / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(cap_utils.to_json_safe(hist_message), ensure_ascii=False) + "\n")
        return {"ok": True, "updated": len(self._history)}

    def set_trace_level(self, level: "cap_utils.TraceLevel | str") -> None:
        """Set the scene history trace level."""
        self._trace_level = cap_utils.TraceLevel(level)

    def serve(self, transport: str = "streamable-http") -> None:
        """Start an MCP server exposing scene-routed agent tools."""
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Serving a scene over MCP requires the mcp package") from exc

        s_config = self._server_config
        self._copy_skills_for_server(s_config)
        server = FastMCP(s_config.cap_id, host=s_config.host, port=s_config.port)
        for method_name in (
            "reset",
            "agent_doc",
            "execute",
            "run_pipe",
            "monitor",
            "retry",
            "record",
            "update_history",
            "get_obs",
        ):
            method = getattr(self, method_name)

            @functools.wraps(method)  # pylint: disable=cell-var-from-loop
            async def _wrapped(*args, _method=method, _name=method_name, **kwargs):
                call_info = [f"MCP scene tool call: {_name}"]
                if args:
                    call_info.append(f"args={cap_utils.summarize_value(args)}")
                if kwargs:
                    call_info.append(f"kwargs={cap_utils.summarize_value(kwargs)}")
                self._logger.info(" ".join(call_info))
                result = _method(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result

            _wrapped.__doc__ = inspect.getdoc(method) or method_name
            _wrapped.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
            server.tool()(_wrapped)

        self._logger.info(
            "Starting scene MCP server %s at http://%s:%s/mcp", s_config.cap_id, s_config.host, s_config.port
        )
        import anyio

        anyio.run(self._run_server_async, server, transport)

    async def _start_task(self, canonical: str, method_name: str, **kwargs: Any) -> dict[str, Any]:
        agent_info = self._agents[canonical]
        task = agent_info.task
        if task is not None and not task.done():
            return cap_utils.to_json_safe(agent_info.status)
        started_time = time.time()
        started_at = datetime.fromtimestamp(started_time).strftime("%Y-%m-%d:%H-%M-%S.%f")[:-3]
        agent_info.status = self._get_status(canonical, method=method_name, running=True, started_at=started_at)
        method = getattr(agent_info.agent, method_name)

        async def _run() -> dict[str, Any]:
            try:
                result = await self._dispatch_task(method, kwargs)
            except BaseException as exc:
                self._logger.exception("Agent task failed: %s.%s", canonical, method_name)
                result = {"ok": False, "error": type(exc).__name__, "err_msg": str(exc)}
            finished_time = time.time()
            status = self._get_status(
                canonical,
                method=method_name,
                started_at=started_at,
                duration=f"{finished_time - started_time:.2f}s",
                result=cap_utils.to_json_safe(result),
            )
            agent_info.status = status
            return status

        if self._config.async_task:
            agent_info.task = asyncio.create_task(_run())
            return cap_utils.to_json_safe(agent_info.status)
        return await _run()

    async def _dispatch_task(self, method: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Dispatch an agent method according to the scene execution mode."""
        if self._config.async_task:
            return await asyncio.to_thread(method, **kwargs)
        return method(**kwargs)

    async def _wait_task(self, canonical: str, wait_ms: int) -> None:
        task = self._agents[canonical].task
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
        return self._agents[self._resolve_name(agent)].agent

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
        """Render scene-bound skills in bundled or prefixed fast mode."""
        source_dir = Path(__file__).resolve().parents[2] / "skills"
        skill_root = Path(server_config.skill_folder).expanduser()
        target_dir = skill_root / server_config.cap_id
        if not skill_root.exists() or not any(skill_root.iterdir()):
            self._logger.info("Skip copying skills because skill_folder is missing or empty: %s", skill_root)
            return target_dir
        if not source_dir.exists():
            self._logger.warning("Skill source directory does not exist: %s", source_dir)
            return target_dir
        replacements = {
            "{cap_id}": server_config.cap_id,
            "{available_names}": ", ".join(sorted(self._agent_aliases)),
        }

        def copy_skill_tree(source: Path, target: Path) -> None:
            if target.resolve() == source.resolve():
                raise ValueError("server.skill_folder target must not point to the source skills directory")
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            for source_path in source.rglob("*"):
                if "__pycache__" in source_path.parts:
                    continue
                target_path = target / source_path.relative_to(source)
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

        if server_config.fast:
            for source_path in source_dir.iterdir():
                if source_path.is_dir() and source_path.name != "__pycache__":
                    copy_skill_tree(source_path, skill_root / f"{server_config.cap_id}_{source_path.name}")
            return target_dir

        copy_skill_tree(source_dir, target_dir)
        return target_dir

    @property
    def server_config(self) -> ServerConfig:
        """Return this scene's MCP server configuration."""
        return self._server_config

    @property
    def agents(self) -> dict[str, BaseAgent]:
        """Return agents keyed by canonical name."""
        return {name: info.agent for name, info in self._agents.items()}
