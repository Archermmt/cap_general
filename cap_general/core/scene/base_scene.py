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
from cap_general.core.control import BaseControl
from cap_general.core.base import RegisteredBase
from cap_general.core.utils.config import build_dataclass_config, load_yaml_config
from cap_general.frameworks import import_frameworks


@dataclass
class ServerConfig:
    """Configuration for the MCP server used by a scene."""

    cap_id: str = "cap"
    host: str = "127.0.0.1"
    port: int = 8080
    fast: bool = True


@dataclass
class ControlInfo:
    """Runtime state for one scene control."""

    control: BaseControl
    status: dict[str, Any]
    task: asyncio.Future[Any] | asyncio.Task[Any] | None = None

    @property
    def agent(self) -> BaseControl:
        """Compatibility alias for old scene internals."""
        return self.control


AgentInfo = ControlInfo


@dataclass
class BaseSceneConfig:
    """Configuration for a scene containing one or more agents."""

    controls: list[dict[str, Any]] | None = None
    agents: list[dict[str, Any]] | None = None
    server: ServerConfig = field(default_factory=ServerConfig)
    record_dir: str | Path = "outputs/scene"
    export_dir: str | Path = "outputs/export"
    debug: bool = False
    async_task: bool = False
    task_async: bool | None = None
    trace_level: cap_utils.TraceLevel | str = cap_utils.TraceLevel.ALL

    def __post_init__(self) -> None:
        if self.task_async is not None:
            self.async_task = bool(self.task_async)


@RegisteredBase.register()
class BaseScene(RegisteredBase):
    """A scene contains multiple named agents."""

    _registry: ClassVar[dict[str, type["BaseScene"]]] = {}
    registry_key_attr: ClassVar[str] = "scene_type"
    scene_type: ClassVar[str] = "base"
    config_cls: ClassVar[type[BaseSceneConfig]] = BaseSceneConfig

    @staticmethod
    def trace_result(method: Callable[..., Any]) -> Callable[..., Any]:
        """Trace one request and response per control routed by a scene method."""
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
            arguments = {key: value for key, value in arguments.items() if value is not None}
            self._append_history(
                {"role": "user", "tool": method.__name__, "request": cap_utils.to_json_safe(arguments)}
            )

        def trace_response(self: "BaseScene", results: dict[str, Any]) -> None:
            for agent_mark, result in results.items():
                redundant_keys = {"agent", "control"}
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
        self._control_aliases: dict[str, str] = {}
        self._controls: dict[str, ControlInfo] = {}
        self._agents = self._controls
        specs = self._config.controls if self._config.controls is not None else self._config.agents
        if specs is None:
            raise ValueError("Scene config requires 'controls' (or legacy 'agents')")
        self._build_controls(specs)

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

    def _build_controls(self, specs: list[dict[str, Any]]) -> None:
        self._pre_build()
        for spec_data in specs:
            raw = dict(spec_data)
            name = raw.pop("name")
            alias = raw.pop("alias", None)
            if name in self._controls:
                raise ValueError(f"Duplicate control name in scene: {name}")
            raw.pop("debug", None)
            raw["record_dir"] = self._record_dir / name
            raw["debug"] = self._config.debug
            raw["name"] = name
            raw["alias"] = alias
            if raw.get("pipeline"):
                pipeline = dict(raw["pipeline"])
                pipeline["export_dir"] = str(self._export_dir / name)
                raw["pipeline"] = pipeline
            control = BaseControl.from_config(raw, logger=self._logger)
            self._controls[name] = ControlInfo(control=control, status=self._get_status(name))
            if alias:
                existing = self._control_aliases.get(alias)
                if existing is not None and existing != name:
                    self._logger.warning(
                        "Skip duplicate control alias %r for %s; already bound to %s", alias, name, existing
                    )
                else:
                    self._control_aliases[alias] = name
        self._post_build()

    def _pre_build(self) -> None:
        """Hook called before agents are constructed."""

    def _post_build(self) -> None:
        """Hook called after all agents are constructed."""
        for control_info in self._controls.values():
            control_info.control.post_build(self)

    @trace_result
    def reset(
        self,
        control_options: dict[str, dict[str, Any]] | None = None,
        agent_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Reset multiple controls from a control-to-options mapping."""
        control_options = control_options if control_options is not None else agent_options
        if control_options is None:
            raise ValueError("reset requires control_options")
        requests = self._resolve_kwargs(control_options)
        results: dict[str, Any] = {}
        for canonical, options in requests.items():
            control = self._get_control(canonical)
            results[control.mark] = control.reset(options=options)
        return results

    @trace_result
    def control_doc(self, controls: list[str] | None = None, agents: list[str] | None = None) -> dict[str, Any]:
        """Return documentation for the selected controls, or all controls if omitted."""
        controls = controls if controls is not None else agents
        if controls is None:
            raise ValueError("control_doc requires controls")
        controls_doc = {}
        for canonical in self._resolve_names(controls):
            control = self._get_control(canonical)
            controls_doc[control.mark] = control.control_doc()
        return {
            "scene": {
                "async_task": self._config.async_task,
                "controls": controls_doc,
            }
        }

    @trace_result
    async def execute(
        self,
        control_codes: dict[str, str] | None = None,
        agent_codes: dict[str, str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Start code execution tasks for each selected control."""
        control_codes = control_codes if control_codes is not None else agent_codes
        if control_codes is None:
            raise ValueError("execute requires control_codes")
        requests = self._resolve_kwargs(control_codes)
        results = [await self._start_task(control, "execute", code=code) for control, code in requests.items()]
        return self._format_results(requests, results)

    @trace_result
    async def retry(self, controls: list[str] | None = None, agents: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Start retry tasks for selected controls."""
        controls = controls if controls is not None else agents
        if controls is None:
            raise ValueError("retry requires controls")
        canonical_controls = self._resolve_names(controls)
        results = [await self._start_task(control, "retry") for control in canonical_controls]
        return self._format_results(canonical_controls, results)

    @trace_result
    def run_pipe(
        self,
        control_options: dict[str, dict[str, Any]] | None = None,
        agent_options: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Run pipeline jobs for selected controls from a control-to-options mapping."""
        control_options = control_options if control_options is not None else agent_options
        if control_options is None:
            raise ValueError("run_pipe requires control_options")
        requests = self._resolve_kwargs(control_options)
        results = [self._run_task_sync(control, "run_pipe", **kwargs) for control, kwargs in requests.items()]
        return self._format_results(requests, results)

    @trace_result
    async def monitor(
        self,
        controls: list[str] | None = None,
        wait_ms: int = -1,
        agents: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return selected controls' execution statuses, optionally waiting for completion."""
        controls = controls if controls is not None else agents
        if controls is None:
            raise ValueError("monitor requires controls")
        canonical_controls = self._resolve_names(controls)
        if wait_ms != 0:
            await asyncio.gather(*(self._wait_task(control, wait_ms) for control in canonical_controls))
        return self._format_results(
            canonical_controls,
            [cap_utils.to_json_safe(self._controls[control].status) for control in canonical_controls],
        )

    @trace_result
    def get_obs(self, controls: list[str] | None = None, agents: list[str] | None = None) -> dict[str, Any]:
        """Return observations for the selected controls, or all controls if omitted."""
        controls = controls if controls is not None else agents
        if controls is None:
            raise ValueError("get_obs requires controls")
        results: dict[str, Any] = {}
        for canonical in self._resolve_names(controls):
            control = self._get_control(canonical)
            results[control.mark] = control.get_obs()
        return results

    @trace_result
    def record(
        self,
        controls: list[str] | None = None,
        clean_frames: bool = False,
        agents: list[str] | None = None,
    ) -> dict[str, Any]:
        """Record complete run artifacts for selected controls, or all controls if omitted."""
        controls = controls if controls is not None else agents
        if controls is None:
            raise ValueError("record requires controls")
        results: dict[str, Any] = {}
        for canonical in self._resolve_names(controls):
            control = self._get_control(canonical)
            results[control.mark] = control.record(step_idx=-1, clean_frames=clean_frames)
        return results

    def update_history(
        self,
        control_messages: dict[str, dict[str, Any]] | None = None,
        agent_messages: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Append one history message per selected control to the scene transcript."""
        control_messages = control_messages if control_messages is not None else agent_messages
        if control_messages is None:
            raise ValueError("update_history requires control_messages")
        requests = self._resolve_kwargs(control_messages)
        results: dict[str, Any] = {}
        for canonical, message in requests.items():
            control = self._get_control(canonical)
            results[control.mark] = self._append_history(message, canonical=canonical)
        return results

    def agent_doc(self, agents: list[str]) -> dict[str, Any]:
        """Compatibility alias for ``control_doc``."""
        result = self.control_doc(controls=agents)
        result["scene"]["agents"] = result["scene"]["controls"]
        return result

    def run_pipe_legacy(self, control_options: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        """Compatibility alias for legacy control_options callers."""
        return self.run_pipe(control_options)

    def update_history_legacy(self, control_messages: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Compatibility alias for legacy control_messages callers."""
        return self.update_history(control_messages)

    def _append_history(self, message: dict[str, Any], canonical: str | None = None) -> dict[str, Any]:
        """Persist one message in the scene history."""
        if not isinstance(message, dict):
            raise TypeError("message must be a history message dictionary")
        hist_message = {
            "timestamp": datetime.now().strftime("%Y-%m-%d:%H-%M-%S.%f")[:-3],
            **message,
        }
        if canonical is not None:
            hist_message["control"] = self._get_control(canonical).mark
        self._history.append(hist_message)
        history_path = self._record_dir / "history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(cap_utils.to_json_safe(hist_message), ensure_ascii=False) + "\n")
        return {"ok": True, "updated": len(self._history)}

    def set_trace_level(self, level: "cap_utils.TraceLevel | str") -> None:
        """Set the scene history trace level."""
        self._trace_level = cap_utils.TraceLevel(level)

    def serve(self, transport: str = "streamable-http", client_type: str = "nanobot") -> None:
        """Start an MCP server exposing scene-routed control tools."""
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Serving a scene over MCP requires the mcp package") from exc

        s_config = self._server_config
        self._copy_skills_for_server(s_config, client_type=client_type)
        server = FastMCP(s_config.cap_id, host=s_config.host, port=s_config.port)
        for method_name in (
            "reset",
            "control_doc",
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
        control_info = self._controls[canonical]
        task = control_info.task
        if task is not None and not task.done():
            return cap_utils.to_json_safe(control_info.status)
        started_time = time.time()
        started_at = datetime.fromtimestamp(started_time).strftime("%Y-%m-%d:%H-%M-%S.%f")[:-3]
        control_info.status = self._get_status(canonical, method=method_name, running=True, started_at=started_at)
        method = getattr(control_info.control, method_name)

        async def _run() -> dict[str, Any]:
            try:
                result = await self._dispatch_task(method, kwargs)
            except BaseException as exc:
                self._logger.exception("Control task failed: %s.%s", canonical, method_name)
                result = {"ok": False, "error": type(exc).__name__, "err_msg": str(exc)}
            finished_time = time.time()
            status = self._get_status(
                canonical,
                method=method_name,
                started_at=started_at,
                duration=f"{finished_time - started_time:.2f}s",
                result=cap_utils.to_json_safe(result),
            )
            control_info.status = status
            return status

        if self._config.async_task:
            control_info.task = asyncio.create_task(_run())
            return cap_utils.to_json_safe(control_info.status)
        return await _run()

    def _run_task_sync(self, canonical: str, method_name: str, **kwargs: Any) -> dict[str, Any]:
        control_info = self._controls[canonical]
        task = control_info.task
        if task is not None and not task.done():
            return cap_utils.to_json_safe(control_info.status)
        started_time = time.time()
        started_at = datetime.fromtimestamp(started_time).strftime("%Y-%m-%d:%H-%M-%S.%f")[:-3]
        control_info.status = self._get_status(canonical, method=method_name, running=True, started_at=started_at)
        method = getattr(control_info.control, method_name)
        try:
            result = method(**kwargs)
        except BaseException as exc:
            self._logger.exception("Control task failed: %s.%s", canonical, method_name)
            result = {"ok": False, "error": type(exc).__name__, "err_msg": str(exc)}
        finished_time = time.time()
        status = self._get_status(
            canonical,
            method=method_name,
            started_at=started_at,
            duration=f"{finished_time - started_time:.2f}s",
            result=cap_utils.to_json_safe(result),
        )
        control_info.status = status
        return cap_utils.to_json_safe(status)

    async def _dispatch_task(self, method: Callable[..., Any], kwargs: dict[str, Any]) -> Any:
        """Dispatch a control method according to the scene execution mode."""
        if self._config.async_task:
            return await asyncio.to_thread(method, **kwargs)
        return method(**kwargs)

    async def _wait_task(self, canonical: str, wait_ms: int) -> None:
        task = self._controls[canonical].task
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
            "control": canonical,
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

    def _get_control(self, control: str | None = None) -> BaseControl:
        """Return a control by name or alias."""
        return self._controls[self._resolve_name(control)].control

    def _get_agent(self, agent: str | None = None) -> BaseControl:
        """Compatibility alias for ``_get_control``."""
        return self._get_control(agent)

    def _resolve_name(self, control: str | None = None) -> str:
        """Return the canonical control name for a name or alias."""
        if control is None:
            if len(self._controls) == 1:
                return next(iter(self._controls))
            raise ValueError("control is required when a scene has multiple controls")
        canonical = self._control_aliases.get(control, control)
        if canonical not in self._controls:
            raise KeyError(f"Unknown control: {control}")
        return canonical

    def _resolve_names(self, controls: list[str] | None = None) -> list[str]:
        """Resolve names and aliases to unique canonical control names."""
        requested = list(self._controls) if controls is None else controls
        canonical_controls: list[str] = []
        for control in requested:
            canonical = self._resolve_name(control)
            if canonical not in canonical_controls:
                canonical_controls.append(canonical)
        return canonical_controls

    def _resolve_kwargs(self, values: dict[str, Any]) -> dict[str, Any]:
        """Resolve mapping keys to canonical names and reject alias collisions."""
        resolved: dict[str, Any] = {}
        for control, value in values.items():
            canonical = self._resolve_name(control)
            if canonical in resolved:
                raise ValueError(f"Duplicate control mapping after alias resolution: {control!r} -> {canonical!r}")
            resolved[canonical] = value
        return resolved

    def _format_results(self, controls: Iterable[str], results: Iterable[Any]) -> dict[str, Any]:
        """Key ordered control results by their human-readable response names."""
        formatted: dict[str, Any] = {}
        for canonical, result in zip(controls, results, strict=True):
            control = self._get_control(canonical)
            formatted[control.mark] = result
        return formatted

    def _copy_skills_for_server(
        self,
        server_config: ServerConfig,
        client_type: str = "nanobot",
        agent_config_path: str | Path | None = None,
    ) -> Path:
        """Render agent-specific scene skills in bundled or prefixed fast mode."""
        skills_dir = Path(__file__).resolve().parents[2] / "skills"
        config_path = Path(agent_config_path) if agent_config_path is not None else skills_dir / "config.yaml"
        config_data = load_yaml_config(config_path)
        agents = config_data.get("agents")
        if not isinstance(agents, dict):
            raise TypeError(f"Skills config must contain an 'agents' mapping: {config_path}")
        if client_type not in agents:
            raise KeyError(
                f"Unknown server client type {client_type!r}. Available agent types: {sorted(agents)}"
            )
        agent_config = agents[client_type]
        if not isinstance(agent_config, dict) or not agent_config.get("skill_folder"):
            raise ValueError(f"Agent {client_type!r} must define 'skill_folder' in {config_path}")
        source_dir = skills_dir / client_type
        skill_root = Path(agent_config["skill_folder"]).expanduser()
        target_dir = skill_root / server_config.cap_id
        if not skill_root.exists():
            self._logger.info("Creating missing skill folder root: %s", skill_root)
            skill_root.mkdir(parents=True, exist_ok=True)
        if not source_dir.exists():
            self._logger.warning("Skill source directory does not exist for agent type %s: %s", client_type, source_dir)
            return target_dir
        replacements = {
            "{cap_id}": server_config.cap_id,
            "{available_names}": ", ".join(sorted(self._control_aliases)),
        }
        skill_variant = "SKILL.async" if self._config.async_task else "SKILL.sync"
        variant_skill_dirs = {"cap_execute"}
        copied_targets: list[Path] = []

        def copy_skill_tree(source: Path, target: Path) -> None:
            if target.resolve() == source.resolve():
                raise ValueError("Agent skill_folder target must not point to the source skills directory")
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            def copy_skill_variant(source_dir: Path, target_dir: Path) -> None:
                variant_source = source_dir / skill_variant
                if not variant_source.exists():
                    raise FileNotFoundError(f"Missing skill variant: {variant_source}")
                content = variant_source.read_text(encoding="utf-8")
                for old, new in replacements.items():
                    content = content.replace(old, new)
                variant_target = target_dir / "SKILL.md"
                self._logger.info(
                    "Copy skill variant %s -> %s for async_task=%s",
                    variant_source,
                    variant_target,
                    self._config.async_task,
                )
                variant_target.write_text(content, encoding="utf-8")

            if source.name in variant_skill_dirs:
                copy_skill_variant(source, target)
            for source_path in source.rglob("*"):
                if "__pycache__" in source_path.parts:
                    continue
                if source_path.name in {"SKILL.async", "SKILL.sync"}:
                    continue
                if source.name in variant_skill_dirs and source_path.name == "SKILL.md":
                    continue
                target_path = target / source_path.relative_to(source)
                if source_path.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                    if source_path.name in variant_skill_dirs:
                        copy_skill_variant(source_path, target_path)
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.name == "SKILL.md":
                    content = source_path.read_text(encoding="utf-8")
                    for old, new in replacements.items():
                        content = content.replace(old, new)
                    target_path.write_text(content, encoding="utf-8")
                else:
                    shutil.copy2(source_path, target_path)
            copied_targets.append(target)

        if server_config.fast:
            self._logger.info(
                "Copying %s skills for cap_id=%s from %s into prefixed folders under %s",
                client_type,
                server_config.cap_id,
                source_dir,
                skill_root,
            )
            for source_path in source_dir.iterdir():
                if source_path.is_dir() and source_path.name != "__pycache__":
                    target_path = skill_root / source_path.name
                    self._logger.info("Copy skill folder %s -> %s", source_path, target_path)
                    copy_skill_tree(source_path, target_path)
            self._logger.info("Finished copying skills for cap_id=%s: %s", server_config.cap_id, copied_targets)
            return skill_root

        self._logger.info(
            "Copying %s skills for cap_id=%s from %s -> %s",
            client_type,
            server_config.cap_id,
            source_dir,
            target_dir,
        )
        copy_skill_tree(source_dir, target_dir)
        self._logger.info("Finished copying skills for cap_id=%s: %s", server_config.cap_id, copied_targets)
        return target_dir

    @property
    def server_config(self) -> ServerConfig:
        """Return this scene's MCP server configuration."""
        return self._server_config

    @property
    def controls(self) -> dict[str, BaseControl]:
        """Return controls keyed by canonical name."""
        return {name: info.control for name, info in self._controls.items()}

    @property
    def agents(self) -> dict[str, BaseControl]:
        """Compatibility alias for ``controls``."""
        return self.controls
