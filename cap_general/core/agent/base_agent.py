"""Base classes for CAP agents."""

import contextlib
import functools
import inspect
import io
import json
import logging
import shutil
import sys
import time
import traceback
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core.base import RegisteredBase
from cap_general.core.env import BaseEnv, BaseEnvConfig
from cap_general.core.policy import BasePolicyConfig
from cap_general.core.utils import ResetFrequency, ResetLevel


class Tee(io.TextIOBase):
    """Stream writes to multiple file-like objects."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


@dataclass
class ServerConfig:
    """Configuration for the MCP server used by an agent."""

    name: str = "mcp"
    host: str = "127.0.0.1"
    port: int = 8080
    skill_folder: str = "skills"


@dataclass
class BaseAgentConfig:
    """Configuration for constructing an agent."""

    env: BaseEnvConfig
    policies: dict[str, BasePolicyConfig]
    server: ServerConfig = field(default_factory=ServerConfig)
    record_dir: str | Path = "agent_record"
    max_steps: int = 5000
    max_retry: int = 5
    reset_frequency: ResetFrequency | str = ResetFrequency.NEVER
    record_execute: bool = True


class BaseAgent(RegisteredBase):
    """Base class for agents."""

    _registry: ClassVar[dict[str, type["BaseAgent"]]] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = BaseAgentConfig
    registry_key_method: ClassVar[str] = "agent_type"

    @classmethod
    def agent_type(cls) -> str:
        """Return the registry key for this agent."""
        return "base_agent"

    def __init__(self, config: BaseAgentConfig):
        """Initialize an agent from config."""
        self._config = config
        self._record_dir = Path(self._config.record_dir)
        self._logger = self._build_logger(self._record_dir)
        self._env: BaseEnv = self._build_env(self._config.env, self._logger)
        self._policies = self._build_policies(self._config.policies, self._logger)
        self._exec_globals: dict[str, Any] = {}
        self._reset_frequency = ResetFrequency(self._config.reset_frequency)
        self._exec_cnt, self._trial_cnt = 0, 0
        self._step_infos, self._step_codes = [], []
        self._plan, self._plan_start = {}, 0

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BaseAgent":
        """Initialize an agent from a yaml config file."""
        data = cls._load_yaml_config(config_path)
        return cls.from_config(data)

    @classmethod
    def get_server_url(cls, config_path: str | Path) -> str:
        """Return the MCP server URL configured by an agent YAML file."""
        data = cls._load_yaml_config(config_path)
        server = data.get("server", {})
        s_config = ServerConfig(**server) if isinstance(server, dict) else server
        return f"http://{s_config.host}:{s_config.port}/mcp"

    @staticmethod
    def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
        """Load an agent YAML config, accepting both top-level and agent-wrapped forms."""
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load agent yaml configs") from exc

        with Path(config_path).open() as file:
            data = yaml.safe_load(file) or {}
        if not isinstance(data, dict):
            raise TypeError("Agent yaml config must contain a mapping at the top level")
        return data.get("agent", data)

    @staticmethod
    def _build_logger(record_dir: Path) -> logging.Logger:
        record_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"cap_general.agent.{record_dir.resolve()}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        log_path = record_dir / "agent.log"
        existing_handler = next(
            (
                handler
                for handler in logger.handlers
                if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_path.resolve()
            ),
            None,
        )
        if existing_handler is None:
            handler = logging.FileHandler(log_path, encoding="utf-8")
            handler.setLevel(logging.INFO)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                )
            )
            logger.addHandler(handler)
        return logger

    @staticmethod
    def _build_env(config: dict[str, Any], logger: logging.Logger) -> BaseEnv:
        return BaseEnv.from_config(config, logger=logger)

    @staticmethod
    def _build_policies(configs: dict[str, dict[str, Any]], logger: logging.Logger) -> dict[str, Any]:
        from cap_general.core.policy import BasePolicy

        return {n: BasePolicy.from_config(c, logger=logger) for n, c in configs.items()}

    def reset(self, options: dict[str, Any] | None = None):
        """Reset the agent, environment, or robot to a requested scope.

        Use this tool before starting a new task, before retrying from a clean
        state, or when the environment needs to be restored.

        Args:
            options: Optional reset options. See ``agent_doc()["reset_rules"]``
                for supported keys.

        Returns:
            A dict with ``ok`` so MCP clients can confirm the reset completed.
        """
        options = dict(options or {})
        self._env.reset(options=options)
        reset_level = ResetLevel(options.get("reset_level", ResetLevel.AGENT))
        if reset_level >= ResetLevel.AGENT:
            self._exec_cnt, self._trial_cnt = 0, 0
            self._step_infos, self._step_codes = [], []
            self._plan, self._plan_start = {}, time.time()
            self._clear_record_dir_contents()
        return {"ok": True}

    def agent_doc(self) -> dict:
        """Return agent instructions and available tool references.

        Call this tool before planning or executing a task. The returned
        information describes callable agent functions, configured policy
        capabilities, and execution rules.

        Returns:
            A dict with ``function_doc`` (str), ``policy_doc`` (dict),
            ``execute_rules`` (str), and ``max_retry`` (int).
        """
        return {
            "function_doc": self._function_doc(),
            "policy_doc": self._policy_doc(),
            "execute_rules": self._execute_rules(),
            "max_retry": self._config.max_retry,
        }

    def execute(self, code: str):
        """Execute Python code as a new agent step.

        The code runs in a persistent namespace containing the low-level
        environment as ``env`` and all functions returned by ``functions()``.
        Use ``agent_doc`` first to inspect the available function signatures,
        policy descriptions, reset rules, and execution rules.

        Args:
            code: Python source code to execute.

        Returns:
            A dict with execution status and artifacts, including ``ok``,
            ``stdout``, ``stderr``, ``result``, ``reward``, ``truncated``,
            ``exec_cnt``, ``trial_cnt``, step range metadata, and ``obs``.
        """
        if self._reset_frequency is ResetFrequency.EXECUTE:
            self.reset(options={"reset_level": ResetLevel.ROBOT})
        self._exec_cnt += 1
        self._trial_cnt = 1
        return self._execute_once(code)

    def retry(self):
        """Retry the most recent ``execute`` call.

        Re-executes the last submitted code without incrementing ``exec_cnt``.
        ``trial_cnt`` is incremented for the retry. If ``max_retry`` has already
        been reached, no code is executed and an error dict is returned.

        Returns:
            The same execution result shape as ``execute`` on success, or a dict
            with ``ok=False`` and ``error="max_retry_exceeded"`` when the retry
            limit is exceeded.
        """
        max_retry = self._config.max_retry
        if self._trial_cnt - 1 >= max_retry:
            return {
                "ok": False,
                "error": "max_retry_exceeded",
                "stderr": f"Exceeded max_retry={max_retry}",
                "exec_cnt": self._exec_cnt,
                "trial_cnt": self._trial_cnt,
                "max_retry": max_retry,
            }
        self._trial_cnt += 1
        return self._execute_once(self._step_codes[-1])

    def record(self, step_idx: int = -1):
        """Persist execution artifacts and return their metadata.

        Args:
            step_idx: Step record index to save. ``-1`` saves the full run,
                including the accumulated plan. Non-negative values save a
                single recorded step/trial.

        Returns:
            A dict containing saved media paths from the environment plus
            ``info`` and ``code`` for the requested scope.
        """
        if step_idx == -1:
            info = {
                "plan": self._plan,
                "executes": self._step_infos,
                "total_execute": len(self._step_infos),
                "total_step": self._env.step_cnt,
                "total_duration": time.time() - self._plan_start,
            }
            code = self._join_codes()
            start_frm, end_frm = 0, self._env.step_cnt
            record_path = self._record_dir
        else:
            info, code = self._get_step_record(step_idx)
            start_frm, end_frm = info["step_start"], info["step_end"]
            record_path = self._record_dir / "step_{}/trial_{}".format(info["exec_cnt"], info["trial_cnt"])
        record_path.mkdir(parents=True, exist_ok=True)
        record = self._env.record(record_path, start_frm=start_frm, end_frm=end_frm)
        self._write_json(record_path / "info.json", info)
        self._write_text(record_path / "code.py", code)
        return {**record, "info": info, "code": code}

    def update_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Merge key-value pairs into the run plan.

        Args:
            plan: Dict of plan fields to merge into the current plan using
                a shallow ``dict.update``.

        Returns:
            The full accumulated plan after the update.
        """
        self._plan.update(plan)
        return self._plan

    def get_obs(self) -> dict[str, Any]:
        """Return the current observation and save images under the active step directory.

        Returns:
            A dict containing the environment observation. Image observations are
            returned as local file paths.
        """
        return self._env.get_observation(self._record_dir / self.step_dir)

    def serve(self, transport: str = "streamable-http") -> None:
        """Start an MCP server for this agent.

        The server is configured by ``self.server_config`` and exposes the
        public agent tools: ``reset``, ``agent_doc``, ``execute``, ``retry``,
        ``record``, ``update_plan``, and ``get_obs``.

        Args:
            transport: MCP transport passed to ``FastMCP.run``. Defaults to
                ``"streamable-http"``.
        """
        try:
            from mcp.server.fastmcp import FastMCP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Serving an agent over MCP requires the mcp package") from exc

        s_config = self._config.server
        server = FastMCP(s_config.name, host=s_config.host, port=s_config.port)
        for method_name in (
            "reset",
            "agent_doc",
            "execute",
            "retry",
            "record",
            "update_plan",
            "get_obs",
        ):
            method = getattr(self, method_name)

            @functools.wraps(method)  # pylint: disable=cell-var-from-loop
            def _wrapped(*args, _method=method, _name=method_name, **kwargs):
                self._logger.info("MCP tool call: %s", _name)
                return _method(*args, **kwargs)

            _wrapped.__doc__ = f"[{s_config.name} only] " + self._mcp_tool_doc(method_name, method)
            _wrapped.__signature__ = inspect.signature(method)  # type: ignore[attr-defined]
            server.tool()(_wrapped)

        self._logger.info(
            "Starting MCP server %s at http://%s:%s/mcp",
            s_config.name,
            s_config.host,
            s_config.port,
        )
        server.run(transport=transport)

    def _mcp_tool_doc(self, method_name: str, method: Callable[..., Any]) -> str:
        """Return the docstring to expose for an MCP tool."""
        if method_name != "reset":
            return inspect.getdoc(method) or ""

        reset_rules = "\n".join(f"        {line}" if line else "" for line in self._reset_rules().splitlines())
        return (
            "Reset the agent, environment, or robot to a requested scope.\n\n"
            "Use this tool before starting a new task, before retrying from a clean "
            "state, or when the environment needs to be restored.\n\n"
            "Args:\n"
            "    options: Optional reset options.\n"
            f"{reset_rules}\n\n"
            "Returns:\n"
            "    A dict with ``ok`` so MCP clients can confirm the reset completed."
        )

    def _execute_once(self, code: str):
        """Execute generated code and return a Gymnasium-style transition tuple."""
        self._clear_current_step_dir()
        if self._reset_frequency is ResetFrequency.TRIAL:
            self.reset(options={"reset_level": ResetLevel.ROBOT})
        step_start, time_start = self._env.step_cnt, time.time()
        exec_result = self._execute_code(code)
        max_steps = self._config.max_steps
        info = {
            **exec_result,
            "step_start": step_start,
            "step_end": self._env.step_cnt,
            "duration": time.time() - time_start,
            "exec_cnt": self._exec_cnt,
            "trial_cnt": self._trial_cnt,
            "reward": self._compute_reward(),
            "truncated": self._env.step_cnt > max_steps,
            "obs": self.get_obs(),
        }
        self._step_infos.append(info)
        self._step_codes.append(code)
        if self._config.record_execute:
            self.record(len(self._step_infos) - 1)
        return info

    def _execute_code(self, code: str) -> dict[str, Any]:
        self._init_exec_globals()
        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                exec(code, self._exec_globals, self._exec_globals)
        except BaseException:
            ok = False
            traceback.print_exc(file=tee_err)
        return {
            "ok": ok,
            "stdout": stdout_buffer.getvalue(),
            "stderr": stderr_buffer.getvalue(),
            "result": self._exec_globals.get("RESULT"),
        }

    def _init_exec_globals(self) -> None:
        """Initialize the persistent globals dictionary for generated code."""
        g: dict[str, Any] = {"__name__": "__main__", "env": self._env, "INPUTS": {}, "RESULT": None}
        for fn_name, fn in self.functions().items():
            g[fn_name] = fn
        self._exec_globals = g

    def _run_policy(self, policy_name: str, method="inference", **kwargs) -> Any:
        """Run a configured policy by name."""
        if policy_name not in self._policies:
            self._logger.warning("Unknown policy requested: %s", policy_name)
            return None
        policy_method = getattr(self._policies[policy_name], method, None)
        if not callable(policy_method):
            self._logger.warning("Policy %r has no callable method %r", policy_name, method)
            return None
        return policy_method(**kwargs)

    def _compute_reward(self) -> float:
        """Compute the current reward."""
        return 0.0

    def _function_doc(self) -> str:
        """Aggregate function docs in a simple, consistent format.

        Format per function:
            name(signature)
              Summary: first line of function doc
              Doc: full function docstring (Google style recommended)
        """
        # we need to discuss this design further down the line
        lines: list[str] = []
        for name, fn in self.functions().items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(…)"
            doc = inspect.getdoc(fn) or ""
            lines.append(f"{name}{sig}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()

    def _policy_doc(self) -> dict[str, dict[str, str]]:
        """Return capability descriptions for configured policies."""
        return {name: policy.describe for name, policy in self._policies.items()}

    def _reset_rules(self) -> str:
        """Return the rules for reset options."""
        return (
            "reset_level: 0 resets only the robot pose, 1 resets the environment, "
            "and 2 resets the full agent state. Defaults to 2."
        )

    def _execute_rules(self) -> str:
        """Return the rules for executing code."""
        return ""

    def _get_step_record(self, step_idx: int) -> tuple[dict[str, Any], str]:
        if step_idx < 0 or step_idx >= len(self._step_infos):
            raise IndexError(f"step_idx {step_idx} out of range for {len(self._step_infos)} records")
        return self._step_infos[step_idx], self._step_codes[step_idx]

    def _join_codes(self) -> str:
        chunks = []
        for info, code in zip(self._step_infos, self._step_codes):
            step_dir = "step_{}/trial_{}".format(info["exec_cnt"], info["trial_cnt"])
            chunks.append(f"# {step_dir}\n{code.rstrip()}\n")
        return "\n".join(chunks)

    def _clear_current_step_dir(self) -> None:
        """Remove artifacts for the current execute/trial slot before writing new ones."""
        self._remove_path(self._record_dir / self.step_dir)

    def _clear_record_dir_contents(self) -> None:
        """Remove all run artifacts under ``record_dir`` after a full agent reset."""
        self._close_record_file_handlers()
        self._record_dir.mkdir(parents=True, exist_ok=True)
        for child in self._record_dir.iterdir():
            self._remove_path(child)
        self._logger = self._build_logger(self._record_dir)

    def _close_record_file_handlers(self) -> None:
        for handler in list(self._logger.handlers):
            if isinstance(handler, logging.FileHandler):
                self._logger.removeHandler(handler)
                handler.close()

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        with path.open("w", encoding="utf-8") as file:
            file.write(text)
            if text and not text.endswith("\n"):
                file.write("\n")

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return mapping of agent function name to callable."""
        raise NotImplementedError

    @property
    def step_dir(self) -> Path:
        """Path to the current step directory."""
        return "step_{}/trial_{}".format(self._exec_cnt, self._trial_cnt)

    @property
    def env(self):
        """Low-level environment instance for direct interaction in code execution."""
        return self._env

    @property
    def policies(self) -> dict[str, Any]:
        """Configured policies keyed by policy name."""
        return self._policies

    @property
    def logger(self) -> logging.Logger:
        """Shared logger used by the agent, environment, and policies."""
        return self._logger
