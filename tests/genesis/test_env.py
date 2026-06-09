"""Tests for CapEnv environment."""

import pytest
from cap_general.core.robot import CapEnv
from cap_general.core.models import StaticPolicyModel, CallablePolicyModel
from cap_general.core.agent import CapStepResult


def test_cap_env_prompt_construction():
    """Test that prompt includes task description and API docs."""

    def simple_generator(prompt: str) -> str:
        return "pass"

    env = CapEnv(
        task_description="Test task",
        api_docs="def test_method():\n    pass",
        policy_model=CallablePolicyModel(simple_generator),
    )

    full_prompt = env._construct_prompt("Additional instruction")

    assert "Test task" in full_prompt
    assert "def test_method():" in full_prompt
    assert "Additional instruction" in full_prompt


def test_cap_env_successful_single_turn():
    """Test successful single-turn execution."""

    fixed_code = "result = 42"
    model = StaticPolicyModel(fixed_code)

    env = CapEnv(
        task_description="Calculate result",
        api_docs="",
        policy_model=model,
        max_steps=1,
    )

    run_result = env.run()

    assert len(run_result.steps) == 1
    assert run_result.steps[0].success is True
    assert run_result.steps[0].generated_code == fixed_code


def test_cap_env_failed_code_execution():
    """Test handling of failed code execution."""

    failing_code = "raise RuntimeError('intentional failure')"
    model = StaticPolicyModel(failing_code)

    env = CapEnv(
        task_description="This will fail", api_docs="", policy_model=model, max_steps=1
    )

    run_result = env.run()

    assert len(run_result.steps) == 1
    assert run_result.steps[0].success is False
    assert "RuntimeError" in run_result.steps[0].execution_result.error


def test_cap_env_multi_turn_stop_on_completion():
    """Test that multi-turn execution stops when done flag is set."""

    step_count = [0]

    def progressive_generator(prompt: str) -> str:
        step_count[0] += 1
        if step_count[0] >= 3:
            # Signal completion on step 3
            return "done = True"
        return f"step = {step_count[0]}"

    model = CallablePolicyModel(progressive_generator)

    env = CapEnv(
        task_description="Multi-step task",
        api_docs="",
        policy_model=model,
        max_steps=10,  # Allow up to 10 steps, but should stop earlier
    )

    run_result = env.run()

    # Should stop at step 3 when done=True is detected
    assert len(run_result.steps) == 3
    assert run_result.steps[2].done is True


def test_cap_env_max_steps_limit():
    """Test that execution stops at max_steps even without done signal."""

    def never_done_generator(prompt: str) -> str:
        return "x = 1"  # Never sets done = True

    model = CallablePolicyModel(never_done_generator)

    env = CapEnv(
        task_description="Long task", api_docs="", policy_model=model, max_steps=5
    )

    run_result = env.run()

    # Should stop at max_steps
    assert len(run_result.steps) == 5


def test_cap_env_reward_tracking():
    """Test that rewards are tracked across steps."""

    def reward_generator(prompt: str) -> str:
        # Simulate increasing reward
        return "reward = 0.5"

    model = CallablePolicyModel(reward_generator)

    env = CapEnv(
        task_description="Reward task", api_docs="", policy_model=model, max_steps=3
    )

    run_result = env.run()

    assert len(run_result.steps) == 3
    assert run_result.final_reward >= 0
