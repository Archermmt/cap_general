"""In-process Python code executor with persistent state."""

import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any
from cap_general.core.result import ExecutionResult


class CodeExecutor:
    """Executes Python code strings in a persistent environment.

    Maintains a globals dictionary that persists across executions,
    allowing variables and imports to carry over between code runs.
    Captures stdout and stderr output.
    """

    def __init__(self):
        """Initialize the executor with a clean global namespace."""
        self.globals: Dict[str, Any] = {
            "__builtins__": __builtins__,
        }

    def run(self, code: str) -> ExecutionResult:
        """Execute a code string and capture results.

        Args:
            code: Python code to execute.

        Returns:
            ExecutionResult with success status, output, and any errors.
        """
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code, self.globals)

            # Execution succeeded
            return ExecutionResult(
                success=True,
                stdout=stdout_buffer.getvalue(),
                stderr=stderr_buffer.getvalue(),
                error=None,
            )

        except Exception as e:
            # Execution failed
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
