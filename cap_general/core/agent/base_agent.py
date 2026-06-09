"""Base classes for CAP agents."""

import io
import inspect
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, ClassVar, Dict

from cap_general.core.base import RegisteredBase
from cap_general.core.agent.result import ExecutionResult


class AgentBase(RegisteredBase):
    """Base class for agents that expose APIs and may execute generated code."""

    _registry: ClassVar[dict[str, type["AgentBase"]]] = {}
    registry_key_method: ClassVar[str] = "agent_type"

    @classmethod
    def agent_type(cls) -> str:
        """Return the registry key for this agent."""
        return cls.__name__

    def combined_doc(self) -> str:
        """Extract and combine documentation from all public methods."""
        docs = []

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue

            sig = inspect.signature(method)
            sig_str = f"{name}{sig}"
            docstring = inspect.getdoc(method)

            doc_section = f"def {sig_str}:"
            if docstring:
                doc_section += f'\n    """{docstring}"""'

            docs.append(doc_section)

        return "\n\n".join(docs)

    def api_spec(self) -> Dict[str, Any]:
        """Get public method specifications as a dictionary."""
        spec = {}

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("_"):
                continue

            sig = inspect.signature(method)
            docstring = inspect.getdoc(method)

            spec[name] = {
                "signature": str(sig),
                "docstring": docstring,
                "parameters": {
                    param_name: {
                        "annotation": (
                            str(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else "Any"
                        ),
                        "default": (
                            str(param.default)
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                    }
                    for param_name, param in sig.parameters.items()
                },
                "return_annotation": (
                    str(sig.return_annotation)
                    if sig.return_annotation != inspect.Signature.empty
                    else "Any"
                ),
            }

        return spec

    def run(self, code: str) -> ExecutionResult:
        """Execute generated code and return the execution result."""
        raise NotImplementedError(f"{type(self).__name__} does not execute code")

    def reset(self):
        """Reset the agent state."""
        pass


@AgentBase.register()
class CodeExecutor(AgentBase):
    """Executes Python code strings in a persistent environment."""

    name = "Code Executor"

    def __init__(self):
        """Initialize the executor with a clean global namespace."""
        self.globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
        }

    @classmethod
    def agent_type(cls) -> str:
        return "code_executor"

    def run(self, code: str) -> ExecutionResult:
        """Execute a code string and capture results."""
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self.globals)

            return ExecutionResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                error=None,
            )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            return ExecutionResult(
                success=False,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                error=error_msg,
            )

    def reset(self):
        """Reset the executor's global namespace."""
        self.globals = {
            "__builtins__": __builtins__,
        }
