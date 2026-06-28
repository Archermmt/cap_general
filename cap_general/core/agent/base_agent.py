"""Base classes for CAP agents."""

import contextlib
import inspect
import io
import logging
import sys
import time
import traceback
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core import utils as cap_utils
from cap_general.core.base import RegisteredBase
from cap_general.core.robot import BaseRobot, BaseRobotConfig
from cap_general.core.policy import BasePolicyConfig


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
class BaseAgentConfig:
    """Configuration for constructing an agent."""

    robot: BaseRobotConfig
    policies: dict[str, BasePolicyConfig] = field(default_factory=dict)
    record_dir: str | Path = "outputs"
    max_steps: int = 5000
    max_retry: int = 5
    reset_mode: cap_utils.ResetMode | str = cap_utils.ResetMode.NEVER
    trace_level: cap_utils.TraceLevel | str = cap_utils.TraceLevel.ALL
    debug: bool = False
    name: str | None = None
    alias: str | None = None


class BaseAgent(RegisteredBase):
    """Base class for agents."""

    _registry: ClassVar[dict[str, type["BaseAgent"]]] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = BaseAgentConfig
    registry_key_method: ClassVar[str] = "agent_type"

    @classmethod
    def agent_type(cls) -> str:
        """Return the registry key for this agent."""
        return "base_agent"

    def __init__(self, config: BaseAgentConfig, logger: logging.Logger):
        """Initialize an agent from config."""
        self._config = config
        self._record_dir = Path(self._config.record_dir).expanduser().resolve()
        self._logger = logger
        self._robot: BaseRobot = self._build_robot(self._config.robot, self._logger)
        self._policies = self._build_policies(self._config.policies, self._logger)
        self._exec_globals: dict[str, Any] = {}
        self._reset_mode = cap_utils.ResetMode(self._config.reset_mode)
        self._exec_cnt, self._trial_cnt = 0, 0
        self._step_infos, self._step_codes = [], []
        self._task_start = 0
        self._history: list[dict[str, Any]] = []
        self._train_epoch = 0
        self._trace_level = cap_utils.TraceLevel(self._config.trace_level)
        self._clear_record_dir_contents()

    @staticmethod
    def _build_robot(config: dict[str, Any], logger: logging.Logger) -> BaseRobot:
        robot = BaseRobot.from_config(config, logger=logger)
        robot.reset()
        return robot

    @staticmethod
    def _build_policies(configs: dict[str, dict[str, Any]], logger: logging.Logger) -> dict[str, Any]:
        from cap_general.core.policy import BasePolicy

        policies = {n: BasePolicy.from_config(c, logger=logger) for n, c in configs.items()}
        for policy in policies.values():
            policy.reset()
        return policies

    def reset(self, options: dict[str, Any] | None = None):
        """Reset the agent, scene, or robot to a requested scope.

        Use this tool before starting a new task, before retrying from a clean
        state, or when the robot needs to be restored.

        Args:
            options: [_options_doc()]

        Returns:
            A dict with ``ok`` so MCP clients can confirm the reset completed.
        """
        options = dict(options or {})
        self._robot.reset(options=options)
        reset_level = cap_utils.ResetLevel(options.get("reset_level", cap_utils.ResetLevel.AGENT))
        if reset_level >= cap_utils.ResetLevel.AGENT:
            self._exec_cnt, self._trial_cnt = 0, 0
            self._step_infos, self._step_codes = [], []
            self._task_start = time.time()
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
        robot as ``robot`` and all functions returned by ``functions()``.
        Use ``agent_doc`` first to inspect the available function signatures,
        policy descriptions, function-specific options, and execution rules.

        Args:
            code: Python source code to execute.

        Returns:
            A dict with execution status and artifacts, including ``ok``,
            ``stdout``, ``stderr``, ``result``, ``reward``, ``truncated``,
            ``exec_cnt``, ``trial_cnt``, step range metadata, and ``obs``.
        """
        if self._reset_mode is cap_utils.ResetMode.PER_EXEC:
            self.reset(options={"reset_level": cap_utils.ResetLevel.ROBOT})
        self._exec_cnt += 1
        self._trial_cnt = 1
        result = self._execute_once(code)
        self._trace_result("execute", result, args={"code": code})
        return result

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
            result = {
                "ok": False,
                "error": "max_retry_exceeded",
                "stderr": f"Exceeded max_retry={max_retry}",
                "exec_cnt": self._exec_cnt,
                "trial_cnt": self._trial_cnt,
                "max_retry": max_retry,
            }
            self._trace_result("retry", result, args={})
            return result
        self._trial_cnt += 1
        result = self._execute_once(self._step_codes[-1])
        self._trace_result("retry", result, args={})
        return result

    def train(
        self,
        policy_name: str,
        epoch: int,
        method: str = "train",
        options: dict[str, Any] | None = None,
    ):
        """Train a configured policy.

        Performs simple bookkeeping (logging and incrementing ``train_epoch``)
        and delegates the actual training logic to :meth:`_train`, which
        subclasses should override.

        Args:
            policy_name: Name of the policy to train, as configured in
                ``policies``.
            epoch: Number of training epochs to run. Must be positive.
            method: Training method/stage identifier, passed through to
                ``_train`` (e.g. ``"rl"`` or ``"bc"``). Defaults to ``"train"``.
            options: [_options_doc()]

        Returns:
            A dict with ``ok`` and the result of ``_train``, plus the updated
            ``train_epoch`` count.
        """
        if epoch <= 0:
            raise ValueError("epoch must be a positive integer")
        options = dict(options or {})
        self._logger.info(
            "Starting train: policy=%s epoch=%s method=%s options=%s", policy_name, epoch, method, options
        )
        self._train_epoch += epoch
        try:
            result = self._train(policy_name=policy_name, epoch=epoch, method=method, options=options)
            response = {"ok": True, "train_epoch": self._train_epoch, "result": result}
        except Exception as exc:
            self._logger.exception("Train failed: policy=%s method=%s", policy_name, method)
            response = {"ok": False, "train_epoch": self._train_epoch, "error": str(exc)}
        self._trace_result(
            "train",
            response,
            args={"policy_name": policy_name, "epoch": epoch, "method": method, "options": options},
        )
        return response

    def _train(self, policy_name: str, epoch: int, method: str, options: dict[str, Any]) -> Any:
        """Hook for subclasses to implement the actual training logic.

        Args:
            policy_name: Name of the policy to train.
            epoch: Number of training epochs to run.
            method: Training method/stage identifier.
            options: Training options/hyperparameters.

        Returns:
            Subclass-defined training result.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement _train")

    def record(self, step_idx: int = -1):
        """Persist execution artifacts and return their metadata.

        Args:
            step_idx: Step record index to save. ``-1`` saves the full run.
                Non-negative values save a single recorded step/trial.

        Returns:
            A dict containing saved media paths from the robot plus
            ``info`` and ``code`` for the requested scope. History is
            persisted separately in ``record_dir/history.jsonl``.
        """
        if step_idx == -1:
            info = {
                "executes": self._step_infos,
                "total_execute": len(self._step_infos),
                "total_step": self._robot.step_cnt,
                "total_duration": self._format_duration(time.time() - self._task_start),
            }
            code = self._join_codes()
            start_frm, end_frm = 0, self._robot.step_cnt
            record_path = self._record_dir
        else:
            info, code = self._get_step_record(step_idx)
            start_frm, end_frm = info["step_start"], info["step_end"]
            record_path = self._record_dir / self._step_dir_name(info["exec_cnt"], info["trial_cnt"])
        record_path.mkdir(parents=True, exist_ok=True)
        record = self._robot.record(record_path, start_frm=start_frm, end_frm=end_frm)
        cap_utils.write_json(record_path / "info.json", info)
        cap_utils.write_text(record_path / "code.py", code)
        return {**record, "info": info, "code": code}

    def update_history(self, message: dict[str, Any]) -> dict[str, Any]:
        """Append one history message to the agent transcript and persist it.

        Each message is appended to ``record_dir/history.jsonl`` immediately so
        the transcript survives crashes without needing an explicit :meth:`record`
        call.

        Args:
            message: One websocket-style transcript message with fields such
                as ``role``, ``mark``, ``request``, or ``response``.

        Returns:
            A compact acknowledgement with the appended message count.
        """
        if not isinstance(message, dict):
            raise TypeError("message must be a history message dictionary")
        self._history.append(message)
        import json as _json

        history_path = self._record_dir / "history.jsonl"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(cap_utils.to_json_safe(message), ensure_ascii=False) + "\n")
        return {"ok": True, "updated": len(self._history)}

    def enable_trace(self, enabled: bool = True) -> None:
        """Enable or disable full trace recording (sets trace_level to ALL or NEVER)."""
        self._trace_level = cap_utils.TraceLevel.ALL if enabled else cap_utils.TraceLevel.NEVER

    def set_trace_level(self, level: "cap_utils.TraceLevel | str") -> None:
        """Set the trace recording level."""
        self._trace_level = cap_utils.TraceLevel(level)

    def _trace_result(self, method_name: str, result: dict[str, Any], args: dict[str, Any] | None = None) -> None:
        level = self._trace_level
        if level is cap_utils.TraceLevel.NEVER:
            return
        mark = f"step_{self._exec_cnt}_trail_{self._trial_cnt}"
        if level is cap_utils.TraceLevel.ALL and args is not None:
            self.update_history(
                {
                    "role": "llm",
                    "mark": mark,
                    "request": {
                        "tool": method_name,
                        "args": cap_utils.to_json_safe(args),
                    },
                }
            )
        self.update_history(
            {
                "role": self.mark,
                "mark": mark,
                "response": {
                    "tool": method_name,
                    "data": cap_utils.to_json_safe(result),
                },
            }
        )

    def get_obs(self) -> dict[str, Any]:
        """Return the current observation and save images under the active step directory.

        Returns:
            A dict containing the robot observation. Image observations are
            returned as local file paths.
        """
        result = self._robot.get_observation(self.step_dir)
        return cap_utils.to_json_safe(result)

    def _execute_once(self, code: str):
        """Execute generated code and return a Gymnasium-style transition tuple."""
        self._clear_current_step_dir()
        if self._reset_mode is cap_utils.ResetMode.PER_TRIAL:
            self.reset(options={"reset_level": cap_utils.ResetLevel.ROBOT})
        step_start, time_start = self._robot.step_cnt, time.time()
        exec_result = self._execute_code(code)
        max_steps = self._config.max_steps
        info = {
            **exec_result,
            "step_start": step_start,
            "step_end": self._robot.step_cnt,
            "duration": self._format_duration(time.time() - time_start),
            "exec_cnt": self._exec_cnt,
            "trial_cnt": self._trial_cnt,
            "reward": self._compute_reward(),
            "truncated": self._robot.step_cnt > max_steps,
            "obs": self.get_obs(),
        }
        self._step_infos.append(info)
        self._step_codes.append(code)
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
        g: dict[str, Any] = {"__name__": "__main__", "robot": self._robot, "INPUTS": {}, "RESULT": None}
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

    def _update_policy(self, policy_name: str, **kwargs: Any) -> Any:
        """Update a configured policy by name."""
        return self._run_policy(policy_name, method="update", **kwargs)

    def _compute_reward(self) -> float:
        """Compute the current reward."""
        return 0.0

    def _function_doc(self) -> str:
        """Aggregate function docs in a simple, consistent format.

        Format per function:
            name(signature)
              Summary: first line of function doc
              Doc: full function docstring (Google style recommended)

        When a function docstring contains ``options: [_options_doc()]`` and the
        agent provides a non-empty options description for that function, the
        placeholder is replaced inline so callers see function-specific option
        rules directly in ``function_doc``. If no options description is
        available, the placeholder line is removed.
        """
        lines: list[str] = []
        all_fns: dict[str, Callable] = {"reset": self.reset, "train": self.train, **self.functions()}
        placeholder = "options: [_options_doc()]"
        for name, fn in all_fns.items():
            try:
                sig = str(inspect.signature(fn))
            except Exception:
                sig = "(…)"
            doc = inspect.getdoc(fn) or ""
            options_doc = self._options_doc(name).strip()
            if placeholder in doc:
                if options_doc:
                    replacement = "options:\n        " + options_doc.replace(
                        chr(10),
                        chr(10) + "        ",
                    )
                    doc = doc.replace(placeholder, replacement)
                else:
                    doc = doc.replace(f"    {placeholder}\n", "")
                    doc = doc.replace(placeholder, "")
            lines.append(f"{name}{sig}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()

    def _options_doc(self, method_name: str) -> str:
        """Return function-specific options documentation for inline substitution."""
        if method_name == "reset":
            return (
                "reset_level: 0 resets only the robot pose, 1 resets the robot "
                "controller, and 2 resets the full agent state. Defaults to 2."
            )
        if method_name == "train":
            return (
                "Method-specific training options. Refer to the concrete agent's "
                "training implementation for supported fields."
            )
        return ""

    def _policy_doc(self) -> dict[str, dict[str, str]]:
        """Return capability descriptions for configured policies."""
        return {name: policy.describe for name, policy in self._policies.items()}

    def _get_step_record(self, step_idx: int) -> tuple[dict[str, Any], str]:
        if step_idx < 0 or step_idx >= len(self._step_infos):
            raise IndexError(f"step_idx {step_idx} out of range for {len(self._step_infos)} records")
        return self._step_infos[step_idx], self._step_codes[step_idx]

    def _join_codes(self) -> str:
        chunks = []
        for info, code in zip(self._step_infos, self._step_codes):
            chunks.append(f"# {self._step_dir_name(info['exec_cnt'], info['trial_cnt'])}\n{code.rstrip()}\n")
        return "\n".join(chunks)

    def _clear_current_step_dir(self) -> None:
        """Remove artifacts for the current execute/trial slot before writing new ones."""
        cap_utils.remove_path(self._current_step_dir_path())

    @staticmethod
    def _step_dir_name(exec_cnt: int, trial_cnt: int) -> str:
        return "step_{}/trial_{}".format(exec_cnt, trial_cnt)

    def _current_step_dir_path(self) -> Path:
        return self._record_dir / self._step_dir_name(self._exec_cnt, self._trial_cnt)

    def _clear_record_dir_contents(self) -> None:
        """Remove all run artifacts under ``record_dir`` after a full agent reset."""
        self._record_dir.mkdir(parents=True, exist_ok=True)
        for child in self._record_dir.iterdir():
            cap_utils.remove_path(child)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        return f"{seconds:.2f}s"

    def _execute_rules(self) -> str:
        """Return generic execution guidance for agents without custom rules."""
        return "Use only documented agent functions, robot methods, and configured policies."

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return mapping of agent function name to callable."""
        raise NotImplementedError

    def _get_sub_dir(self, *parts: str | Path) -> Path:
        path = self._record_dir.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def step_dir(self) -> Path:
        """Path to the current step directory."""
        return self._get_sub_dir(self._step_dir_name(self._exec_cnt, self._trial_cnt))

    @property
    def debug_dir(self) -> Path:
        """Path to the debug directory for the current step."""
        return self._get_sub_dir(self._step_dir_name(self._exec_cnt, self._trial_cnt), "debug")

    @property
    def train_dir(self) -> Path:
        """Path to the train output directory."""
        return self._get_sub_dir("train")

    @property
    def mark(self) -> str:
        """Scene-visible mark such as ``alias(name)``."""
        cfg_name = self._config.name
        cfg_alias = self._config.alias
        if cfg_alias and cfg_alias == cfg_name:
            cfg_alias = None
        name = cfg_name or self.name
        return f"{cfg_alias}({name})" if cfg_alias else name

    @property
    def policies(self) -> dict[str, Any]:
        """Configured policies keyed by policy name."""
        return self._policies

    @property
    def logger(self) -> logging.Logger:
        """Shared logger used by the agent, robot, and policies."""
        return self._logger
