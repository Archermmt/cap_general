"""Base classes for robot/environment control loops."""

from abc import abstractmethod
from typing import ClassVar, List, Optional

from cap_general.core.agent import CodeExecutor
from cap_general.core.agent.result import CapRunResult, CapStepResult
from cap_general.core.base import RegisteredBase
from cap_general.core.models import PolicyModel


class RobotBase(RegisteredBase):
    """Abstract base class for robot/environment controllers."""

    _registry: ClassVar[dict[str, type["RobotBase"]]] = {}
    registry_key_method: ClassVar[str] = "robot_type"

    @classmethod
    @abstractmethod
    def robot_type(cls) -> str:
        """Return the registry key for this robot controller."""
        pass

    @abstractmethod
    def step(self, step_number: int, additional_instruction: str = "") -> CapStepResult:
        """Execute one robot control step."""
        pass

    @abstractmethod
    def run(self, initial_instruction: str = "") -> CapRunResult:
        """Run a full robot control episode."""
        pass

    @abstractmethod
    def reset(self):
        """Reset the robot/environment state."""
        pass


@RobotBase.register()
class CapEnv(RobotBase):
    """Environment for executing CAP task loops."""

    name = "CAP Environment"

    def __init__(
        self,
        task_description: str,
        api_docs: str,
        policy_model: PolicyModel,
        max_steps: int = 10,
        executor: Optional[CodeExecutor] = None,
    ):
        """Initialize the CAP environment."""
        self.task_description = task_description
        self.api_docs = api_docs
        self.policy_model = policy_model
        self.max_steps = max_steps
        self.executor = executor or CodeExecutor()

    @classmethod
    def robot_type(cls) -> str:
        return "cap_env"

    def _construct_prompt(self, additional_instruction: str = "") -> str:
        """Construct the full prompt for the policy model."""
        prompt_parts = [
            "Task Description:",
            self.task_description,
            "",
            "Available APIs:",
            self.api_docs,
            "",
        ]

        if additional_instruction:
            prompt_parts.extend(["Current Instruction:", additional_instruction, ""])

        prompt_parts.append(
            "Generate Python code to accomplish this task. "
            "Set 'done = True' when the task is complete."
        )

        return "\n".join(prompt_parts)

    def _check_done(self, code: str) -> bool:
        """Check if the generated code signals completion."""
        return "done = True" in code

    def _extract_reward(self) -> float:
        """Extract reward value from executor globals."""
        return self.executor.globals.get("reward", 0.0)

    def step(self, step_number: int, additional_instruction: str = "") -> CapStepResult:
        """Execute a single step in the CAP loop."""
        prompt = self._construct_prompt(additional_instruction)
        generation_result = self.policy_model.generate(prompt)
        generated_code = generation_result.code
        execution_result = self.executor.run(generated_code)
        done = self._check_done(generated_code)
        reward = self._extract_reward()

        return CapStepResult(
            step_number=step_number,
            prompt=prompt,
            generated_code=generated_code,
            execution_result=execution_result,
            done=done,
            reward=reward,
        )

    def run(self, initial_instruction: str = "") -> CapRunResult:
        """Run the complete CAP task loop."""
        steps: List[CapStepResult] = []

        for step_num in range(1, self.max_steps + 1):
            instruction = initial_instruction if step_num == 1 else ""
            step_result = self.step(step_num, instruction)
            steps.append(step_result)

            if step_result.done:
                break

            if not step_result.success:
                break

        final_reward = steps[-1].reward if steps else 0.0
        success = len(steps) > 0 and steps[-1].done and steps[-1].success

        return CapRunResult(
            steps=steps,
            total_steps=len(steps),
            final_reward=final_reward,
            success=success,
        )

    def reset(self):
        """Reset the environment state."""
        self.executor.reset()
