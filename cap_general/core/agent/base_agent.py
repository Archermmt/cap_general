"""Base classes for CAP agents."""

import contextlib
import inspect
import io
import sys
import traceback
from abc import abstractmethod
from collections.abc import Callable
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


class AgentBase(RegisteredBase):
    """Base class for agents."""

    _registry: ClassVar[dict[str, type["AgentBase"]]] = {}
    registry_key_method: ClassVar[str] = "agent_type"

    @classmethod
    def agent_type(cls) -> str:
        """Return the registry key for this agent."""
        return cls.__name__

    def reset(self):
        """Reset the agent state."""
        pass

    def execute(self, code: str):
        """Execute generated code and return a Gymnasium-style transition tuple."""
        exec_result = self._exec_code(code)
        obs = self.env.get_observation()
        reward = self.compute_reward()
        terminated = reward == 1.0
        truncated = self.env.step_cnt > self.max_steps
        info = {
            "ok": exec_result["ok"],
            "stdout": exec_result["stdout"],
            "stderr": exec_result["stderr"],
        }
        return obs, reward, bool(terminated), bool(truncated), info

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
            # first = doc.splitlines()[0] if doc else ""
            lines.append(f"{name}{sig}")
            # if first:
            #     lines.append(f"  Summary: {first}")
            if doc:
                lines.append("  Doc:")
                lines.extend(f"    {ln}" for ln in doc.splitlines())
            lines.append("")
        return "\n".join(lines).strip()

    def _exec_code(self, code: str) -> dict[str, Any]:
        obs = self._get_observation()
        self._exec_globals["obs"] = obs
        self._exec_globals["env"] = self._env
        for fn_name, fn in self.functions().items():
            self._exec_globals[fn_name] = fn

        stdout_buffer = io.StringIO()
        tee_out = Tee(sys.stdout, stdout_buffer)
        stderr_buffer = io.StringIO()
        tee_err = Tee(sys.stderr, stderr_buffer)
        ok = True
        try:
            with (
                contextlib.redirect_stdout(tee_out),
                contextlib.redirect_stderr(tee_err),
            ):
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
    def env(self):
        """Low-level environment instance for direct interaction in code execution."""
        return self._env
