"""Result types for CAP execution."""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class ExecutionResult:
    """Result from executing a single code snippet."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    error: Optional[str] = None


@dataclass
class CapStepResult:
    """Result from a single step in the CAP environment loop."""

    step_number: int
    prompt: str
    generated_code: str
    execution_result: ExecutionResult
    done: bool = False
    reward: float = 0.0

    @property
    def success(self) -> bool:
        """Whether this step executed successfully."""
        return self.execution_result.success


@dataclass
class CapRunResult:
    """Complete result from running a CAP task."""

    steps: List[CapStepResult]
    total_steps: int
    final_reward: float = 0.0
    success: bool = False

    @property
    def last_step(self) -> Optional[CapStepResult]:
        """Get the last step if any steps were executed."""
        return self.steps[-1] if self.steps else None
