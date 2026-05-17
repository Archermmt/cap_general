"""CAP Environment for running task loops with policy models."""

from typing import Optional, List
from cap_general.core.models import PolicyModel
from cap_general.core.executor import CodeExecutor
from cap_general.core.result import CapStepResult, CapRunResult


class CapEnv:
    """Environment for executing CAP task loops.

    Manages the interaction between a policy model, code executor, and task.
    Supports multi-turn execution with automatic stopping on completion.
    """

    def __init__(
        self,
        task_description: str,
        api_docs: str,
        policy_model: PolicyModel,
        max_steps: int = 10,
        executor: Optional[CodeExecutor] = None,
    ):
        """Initialize the CAP environment.

        Args:
            task_description: Natural language description of the task.
            api_docs: Documentation for available APIs (from CapApiBase.combined_doc()).
            policy_model: Policy model for generating code from prompts.
            max_steps: Maximum number of execution steps.
            executor: Optional code executor (creates new one if not provided).
        """
        self.task_description = task_description
        self.api_docs = api_docs
        self.policy_model = policy_model
        self.max_steps = max_steps
        self.executor = executor or CodeExecutor()

    def _construct_prompt(self, additional_instruction: str = "") -> str:
        """Construct the full prompt for the policy model.

        Args:
            additional_instruction: Additional step-specific instruction.

        Returns:
            Complete prompt string.
        """
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
        """Check if the generated code signals completion.

        Args:
            code: Generated code to check.

        Returns:
            True if code contains 'done = True'.
        """
        return "done = True" in code

    def _extract_reward(self) -> float:
        """Extract reward value from executor globals.

        Returns:
            Reward value (default 0.0 if not set).
        """
        return self.executor.globals.get("reward", 0.0)

    def step(self, step_number: int, additional_instruction: str = "") -> CapStepResult:
        """Execute a single step in the CAP loop.

        Args:
            step_number: Current step number (for tracking).
            additional_instruction: Optional step-specific instruction.

        Returns:
            CapStepResult with execution results.
        """
        # Construct prompt
        prompt = self._construct_prompt(additional_instruction)

        # Generate code from policy model
        generation_result = self.policy_model.generate(prompt)
        generated_code = generation_result.code

        # Execute the generated code
        execution_result = self.executor.run(generated_code)

        # Check for completion
        done = self._check_done(generated_code)

        # Extract reward
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
        """Run the complete CAP task loop.

        Args:
            initial_instruction: Optional initial instruction for first step.

        Returns:
            CapRunResult with all steps and final results.
        """
        steps: List[CapStepResult] = []

        for step_num in range(1, self.max_steps + 1):
            # Determine instruction for this step
            instruction = initial_instruction if step_num == 1 else ""

            # Execute step
            step_result = self.step(step_num, instruction)
            steps.append(step_result)

            # Check if we should stop
            if step_result.done:
                break

            # Stop if execution failed
            if not step_result.success:
                break

        # Calculate final reward (use last step's reward or cumulative)
        final_reward = steps[-1].reward if steps else 0.0

        # Determine overall success
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
