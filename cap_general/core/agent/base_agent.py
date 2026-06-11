"""Base classes for CAP agents."""

import contextlib
import inspect
import io
import json
import sys
import time
import traceback
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core.base import RegisteredBase


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

    env: dict[str, Any]
    policys: dict[str, dict[str, Any]] = field(default_factory=dict)
    record_dir: str | Path = "agent_record"
    max_steps: int = 999999


class BaseAgent(RegisteredBase):
    """Base class for agents."""

    _registry: ClassVar[dict[str, type["BaseAgent"]]] = {}
    config_cls: ClassVar[type[BaseAgentConfig]] = BaseAgentConfig
    registry_key_method: ClassVar[str] = "agent_type"

    @classmethod
    def agent_type(cls) -> str:
        """Return the registry key for this agent."""
        return "base_agent"

    def __init__(self, config: BaseAgentConfig | dict[str, Any]):
        """Initialize an agent from config."""
        config_obj = self._load_config(config=config)
        self._env = self._build_env(config_obj.env)
        self._policies = self._build_policys(config_obj.policys)
        self._exec_globals: dict[str, Any] = {}
        self._max_steps = config_obj.max_steps
        self._record_dir = Path(config_obj.record_dir)
        self._exec_cnt, self._trial_cnt = 0, 0
        self._step_infos, self._step_codes = [], []
        self._exec_start: float | None = None

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "BaseAgent":
        """Initialize an agent from a yaml config file."""
        return cls(config=cls._load_yaml(config_path))

    @classmethod
    def _load_config(
        cls,
        config: BaseAgentConfig | dict[str, Any],
    ) -> BaseAgentConfig:
        data: dict[str, Any] = {}
        if is_dataclass(config):
            data.update(config.__dict__)
        elif isinstance(config, dict):
            data.update(config)
        else:
            raise TypeError(f"Unsupported config type: {type(config).__name__}")
        return cls.config_cls(**data)

    @staticmethod
    def _load_yaml(config_path: str | Path) -> dict[str, Any]:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to load agent yaml configs") from exc

        with Path(config_path).open() as file:
            data = yaml.safe_load(file) or {"agent": {}}
        if not isinstance(data, dict) or "agent" not in data:
            raise TypeError("Agent yaml config must contain a mapping at the top level")
        return data["agent"]

    @staticmethod
    def _build_env(config: dict[str, Any] | None):
        if config is None:
            return None
        from cap_general.core.env import BaseEnv

        return BaseEnv.from_config(config)

    @staticmethod
    def _build_policys(configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
        from cap_general.core.policy import PolicyBase

        return {
            policy_name: PolicyBase.from_config(policy_config)
            for policy_name, policy_config in configs.items()
        }

    def reset(self):
        """Reset the agent state."""
        if self._env is not None:
            self._env.reset()
        self._exec_cnt, self._trial_cnt = 0, 0
        self._step_infos, self._step_codes = [], []
        self._exec_start = time.time()

    def get_observation(self) -> Any:
        """Get the current observation from the configured environment."""
        if self._env is None:
            return None
        return self._env.get_observation(images_only=True)

    def describe_agent(self) -> dict:
        """Return the doc for cap-as-policy."""
        return {"combined_doc": self.combined_doc(), "exec_prompt": self.exec_prompt}

    def run_policy(self, policy_name: str, method="generate", **kwargs: Any) -> Any:
        """Run a configured policy by name."""
        if policy_name not in self._policies:
            raise KeyError(f"Unknown policy: {policy_name}")
        policy = self._policies[policy_name]
        policy_method = getattr(policy, method, None)
        if not callable(policy_method):
            raise AttributeError(f"Policy {policy_name!r} has no callable method {method!r}")
        return policy_method(**kwargs)

    def execute(self, code: str):
        """Execute generated code and return a Gymnasium-style transition tuple."""
        self._exec_cnt += 1
        self._trial_cnt = 1
        return self._execute_once(code)

    def retry(self):
        """Retry the last execution."""
        self._trial_cnt += 1
        return self._execute_once(self._exec_codes[-1])

    def _execute_once(self, code: str):
        """Execute generated code and return a Gymnasium-style transition tuple."""
        assert self._env is not None, "Environment must be configured to execute code"
        step_start, time_start = self._env.step_cnt, time.time()
        exec_result = self._exec_code(code)
        info = {
            **exec_result,
            "step_start": step_start,
            "step_end": self._env.step_cnt,
            "duration": time.time() - time_start,
            "exec_cnt": self._exec_cnt,
            "trial_cnt": self._trial_cnt,
            "reward": self.compute_reward(),
            "truncated": self._env.step_cnt > self._max_steps,
        }
        self._step_infos.append(info)
        self._step_codes.append(code)
        return info

    def record(self, step_idx: int = -1):
        """Record the current agent state to disk."""
        if step_idx == -1:
            info = {
                "steps": self._step_infos,
                "total_step": self._env.step_cnt,
                "total_duration": time.time() - self._exec_start,
            }
            code = self._join_codes()
            start_frm, end_frm = 0, self._env.step_cnt
            record_path = self._record_dir
        else:
            info, code = self._get_step_record(step_idx)
            start_frm, end_frm = info["step_start"], info["step_end"]
            record_path = self._record_dir / self.step_dir
        record_path.mkdir(parents=True, exist_ok=True)
        record = self._env.record(record_path, start_frm=start_frm, end_frm=end_frm)
        self._write_json(record_path / "info.json", info)
        self._write_text(record_path / "code.py", code)
        return {**record, "info": info, "code": code}

    def _get_step_record(self, step_idx: int) -> tuple[dict[str, Any], str]:
        if step_idx < 0 or step_idx >= len(self._step_infos):
            raise IndexError(
                f"step_idx {step_idx} out of range for {len(self._step_infos)} records"
            )
        return self._step_infos[step_idx], self._step_codes[step_idx]

    def _join_codes(self) -> str:
        chunks = []
        for info, code in zip(self._step_infos, self._step_codes):
            step_dir = "step_{}/trial_{}".format(info["exec_cnt"], info["trial_cnt"])
            chunks.append(f"# {step_dir}\n{code.rstrip()}\n")
        return "\n".join(chunks)

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

    def combined_doc(self) -> str:
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

    def _exec_code(self, code: str) -> dict[str, Any]:
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
        g: dict[str, Any] = {
            "__name__": "__main__",
            "obs": self._env.get_observation(),
            "env": self._env,
            "INPUTS": {},
            "RESULT": None,
        }
        for fn_name, fn in self.functions().items():
            g[fn_name] = fn
        self._exec_globals = g

    def compute_reward(self) -> float:
        """Compute the current reward."""
        return 0.0

    @abstractmethod
    def functions(self) -> dict[str, Callable[..., Any]]:
        """Return mapping of agent function name to callable."""
        raise NotImplementedError

    @property
    def exec_prompt(self) -> str:
        """Return the task prompt"""
        return ""

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
