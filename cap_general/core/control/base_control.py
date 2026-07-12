"""Base classes for CAP controls."""

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
from cap_general.core.pipeline import BasePipeline
from cap_general.core.policy import BasePolicy
from cap_general.core.robot import BaseRobot, BaseRobotConfig


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
class BaseControlConfig:
    """Configuration for constructing a control."""

    robot: BaseRobotConfig
    policies: dict[str, Any] = field(default_factory=dict)
    pipeline: dict[str, Any] = field(default_factory=dict)
    name: str | None = None
    alias: str | None = None
    record_dir: str | Path = "outputs"
    max_steps: int = 5000
    max_retry: int = 5
    debug: bool = False
    reset_mode: cap_utils.ResetMode | str = cap_utils.ResetMode.PER_EXEC
    record_execute: bool = True


class BaseControl(RegisteredBase):
    """Base class for controls."""

    _registry: ClassVar[dict[str, type["BaseControl"]]] = {}
    config_cls: ClassVar[type[BaseControlConfig]] = BaseControlConfig
    registry_key_attr: ClassVar[str] = "control_type"
    control_type: ClassVar[str] = "base"

    def __init__(self, config: BaseControlConfig, logger: logging.Logger):
        """Initialize a control from config."""
        self._config, self._logger = config, logger
        self._record_dir = Path(self._config.record_dir).expanduser().resolve()
        self._robot: BaseRobot = BaseRobot.from_config(self._config.robot, logger=self._logger)

        # Policies are always present; key is used as policy name if not set.
        self._policies: dict[str, BasePolicy] = {}
        for policy_name, policy_cfg in (config.policies or {}).items():
            cfg = dict(policy_cfg)
            cfg.setdefault("name", policy_name)
            self._policies[policy_name] = BasePolicy.from_config(cfg, logger=self._logger)

        # Pipeline is optional — only needed for production jobs (training etc.).
        self._pipeline: BasePipeline | None = (
            BasePipeline.from_config(config.pipeline, logger=self._logger) if config.pipeline else None
        )

        self._exec_globals: dict[str, Any] = {}
        self._exec_cnt, self._trial_cnt = 0, 0
        self._step_infos, self._step_codes = [], []
        self._task_start = 0
        self._reset_mode = cap_utils.ResetMode(self._config.reset_mode)
        self._clear_record_dir_contents()

    def post_build(self, scene: Any) -> None:
        """Initialize the robot and policies after the scene is built."""
        self._robot.post_build(scene)
        self._robot.reset()
        for policy in self._policies.values():
            policy.reset()
            try:
                policy.visualize(self.viz_dir)
            except Exception as exc:
                self._logger.warning("Skip policy DAG visualization for %s: %s", policy.name, exc)

    def reset(self, options: dict[str, Any] | None = None):
        """Reset the control, scene, or robot to a requested scope.

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
        return {"ok": True}

    def control_doc(self) -> dict:
        """Return control instructions and available tool references.

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

    def agent_doc(self) -> dict:
        """Compatibility alias for ``control_doc``."""
        return self.control_doc()

    def execute(self, code: str):
        """Execute Python code as a new agent step.

        The code runs in a persistent namespace containing the low-level
        robot as ``robot`` and all functions returned by ``functions()``.
        Use ``control_doc`` first to inspect the available function signatures,
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
            result = {
                "ok": False,
                "error": "max_retry_exceeded",
                "stderr": f"Exceeded max_retry={max_retry}",
                "exec_cnt": self._exec_cnt,
                "trial_cnt": self._trial_cnt,
                "max_retry": max_retry,
            }
            return result
        self._trial_cnt += 1
        return self._execute_once(self._step_codes[-1])

    def run_pipe(
        self,
        job_options: list[dict[str, Any]],
        policy_name: str = "policy::base",
    ) -> dict[str, Any]:
        """Run a sequence of pipeline jobs in order.

        Args:
            job_options: Ordered list of job entries.  Each entry is a dict
                with keys ``job`` (job group name) and ``options`` (runtime
                options for that job).
                Example: ``[{"job": "train", "options": {"epoch": 100}}]``.
                Per-job available options:
                [_options_doc()]
            policy_name: Policy key to operate on (default ``"policy::base"``).

        Returns:
            A dict with ``ok`` and per-job ``report``.
        """
        if self._pipeline is None:
            raise ValueError("No pipeline configured for this agent")
        policy = self._get_policy(policy_name)
        actual_name = policy_name if policy_name in self._policies else next(iter(self._policies))
        jobs = [jo["job"] for jo in job_options]
        options = {jo["job"]: jo.get("options", {}) for jo in job_options}
        try:
            new_policy, report = self._pipeline.execute(jobs, policy, self._robot, options)
            new_policy.reset()
            self._policies[actual_name] = new_policy
            self.reset(options={"reset_level": cap_utils.ResetLevel.AGENT})
        except Exception as exc:
            self._logger.exception("run_pipe failed")
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "report": report}

    def record(self, step_idx: int = -1, clean_frames: bool = False):
        """Persist execution artifacts and return their metadata.

        Args:
            step_idx: Step record index to save. ``-1`` saves the full run.
                Non-negative values save a single recorded step/trial.

        Returns:
            A dict containing saved media paths from the robot plus
            ``info`` and ``code`` for the requested scope. Scene history is
            persisted separately under the scene ``record_dir``.
        """
        if step_idx == -1:
            info = {
                "executes": self._step_infos,
                "total_execute": len(self._step_infos),
                "total_step": self._robot.step_cnt,
                "total_duration": f"{time.time() - self._task_start:.2f}s",
            }
            code = "\n".join(
                f"# {self._step_dir_name(info['exec_cnt'], info['trial_cnt'])}\n{code.rstrip()}\n"
                for info, code in zip(self._step_infos, self._step_codes, strict=False)
            )
            start_frm, end_frm = 0, self._robot.step_cnt
            record_path = self._record_dir
        else:
            if step_idx < 0 or step_idx >= len(self._step_infos):
                raise IndexError(f"step_idx {step_idx} out of range for {len(self._step_infos)} records")
            info, code = self._step_infos[step_idx], self._step_codes[step_idx]
            start_frm, end_frm = info["step_start"], info["step_end"]
            record_path = self._record_dir / self._step_dir_name(info["exec_cnt"], info["trial_cnt"])
        record_path.mkdir(parents=True, exist_ok=True)
        record = self._robot.record(record_path, start_frm=start_frm, end_frm=end_frm)
        cap_utils.write_json(record_path / "info.json", info)
        cap_utils.write_text(record_path / "code.py", code)
        if clean_frames:
            self._robot.clean_frames()
        return {**record, "info": info, "code": code}

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
        cap_utils.remove_path(self._current_step_dir_path())
        if self._reset_mode is cap_utils.ResetMode.PER_TRIAL:
            self.reset(options={"reset_level": cap_utils.ResetLevel.ROBOT})
        step_start, time_start = self._robot.step_cnt, time.time()
        exec_result = self._execute_code(code)
        info = {
            **exec_result,
            "step_start": step_start,
            "step_end": self._robot.step_cnt,
            "duration": f"{time.time() - time_start:.2f}s",
            "exec_cnt": self._exec_cnt,
            "trial_cnt": self._trial_cnt,
            "reward": self._compute_reward(),
            "obs": self.get_obs(),
        }
        self._step_infos.append(info)
        self._step_codes.append(code)
        if self._config.record_execute:
            self.record(step_idx=len(self._step_infos) - 1)
        return info

    def _execute_code(self, code: str) -> dict[str, Any]:
        g: dict[str, Any] = {"__name__": "__main__", "robot": self._robot, "INPUTS": {}, "RESULT": None}
        for fn_name, fn in self.functions().items():
            g[fn_name] = fn
        self._exec_globals = g
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

    def _get_policy(self, policy_name: str) -> BasePolicy:
        """Return the named policy, or the only policy if there is exactly one."""
        if len(self._policies) == 1:
            return next(iter(self._policies.values()))
        policy = self._policies.get(policy_name)
        if policy is None:
            raise ValueError(f"Policy {policy_name!r} not found. Available: {list(self._policies)}")
        return policy

    def _run_policy(
        self,
        policy_name: str = "policy::base",
        stage: str = "inference",
        inputs: dict[str, Any] | None = None,
    ) -> Any:
        """Run *stage* on the named policy and return the output."""
        policy = self._get_policy(policy_name)
        try:
            result = policy.run(stage, inputs or {})
            if not result.success:
                return None
            return result.output
        except Exception as exc:
            self._logger.warning("Policy %r stage %r failed: %s", policy_name, stage, exc)
            return None

    def _compute_reward(self) -> float:
        """Compute the current reward."""
        return self._robot._last_reward

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
        base_fns: dict[str, Callable] = {"reset": self.reset}
        if self._pipeline is not None:
            base_fns["run_pipe"] = self.run_pipe
        all_fns: dict[str, Callable] = {**base_fns, **self.functions()}
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
        if method_name == "run_pipe":
            return self._pipeline.options_doc() if self._pipeline is not None else ""
        return ""

    def _policy_doc(self) -> dict[str, dict[str, str]]:
        """Return capability descriptions for configured policies."""
        return {name: policy.describe for name, policy in self._policies.items()}

    @staticmethod
    def _step_dir_name(exec_cnt: int, trial_cnt: int) -> str:
        return f"step_{exec_cnt}/trial_{trial_cnt}"

    def _current_step_dir_path(self) -> Path:
        return self._record_dir / self._step_dir_name(self._exec_cnt, self._trial_cnt)

    def _clear_record_dir_contents(self) -> None:
        """Remove all run artifacts under ``record_dir`` after a full agent reset."""
        self._record_dir.mkdir(parents=True, exist_ok=True)
        for child in self._record_dir.iterdir():
            cap_utils.remove_path(child)

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
    def viz_dir(self) -> Path:
        """Path to the visualization output directory."""
        return self._get_sub_dir("visualize")

    @property
    def mark(self) -> str:
        """Scene-visible mark such as ``alias(name)``."""
        cfg_name = self._config.name
        cfg_alias = self._config.alias
        if cfg_alias and cfg_alias == cfg_name:
            cfg_alias = None
        name = cfg_name or type(self).__name__
        return f"{cfg_alias}({name})" if cfg_alias else name

    @property
    def policies(self) -> dict[str, BasePolicy]:
        """Configured policies keyed by policy name."""
        return self._policies

    @property
    def logger(self) -> logging.Logger:
        """Shared logger used by the agent, robot, and policies."""
        return self._logger
