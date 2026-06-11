"""Tests for core CAP agent, env, and policy primitives."""

import pytest
from cap_general.core.agent import BaseAgent
from cap_general.core.policy import (
    CallablePolicy,
    GraspNetPolicy,
    PolicyBase,
    PyrokiPolicy,
    SAM3Policy,
    StaticPolicy,
    VLLMPolicy,
)
from cap_general.core.env import BaseEnv


@BaseEnv.register()
class CoreDummyEnv(BaseEnv):
    """Small env for core agent tests."""

    name = "Core Dummy Env"

    @classmethod
    def env_type(cls) -> str:
        return "core_dummy"

    def _reset(self, *, seed=None, options=None):
        return self.get_observation(), {"seed": seed, "options": options or {}}

    def _step(self, action):
        return self.get_observation(), 0.0, False, False, {"action": action}

    def get_observation(self):
        return {"step": self.step_cnt}


class SimpleAgent(BaseAgent):
    """A simple test agent."""

    def functions(self):
        """Return functions exposed by this agent."""
        return {
            "add": self.add,
            "multiply": self.multiply,
        }

    def add(self, a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of a and b
        """
        return a + b

    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        return x * y


def test_agent_combined_doc():
    """Test that combined_doc extracts method signatures and docstrings."""
    agent = SimpleAgent(config={"env": {"type": "core_dummy"}})
    doc = agent.combined_doc()

    assert "add" in doc
    assert "multiply" in doc
    assert "Add two numbers" in doc
    assert "Multiply two numbers" in doc
    assert "a: int" in doc or "a:int" in doc
    assert "b: int" in doc or "b:int" in doc


def test_static_policy_inference():
    """Test static policy returns fixed code."""
    fixed_code = "action = [1.0, 2.0, 3.0]"
    model = StaticPolicy(code=fixed_code)
    result = model.inference(prompt="test prompt")

    assert result.code == fixed_code
    assert result.policy_name == "StaticPolicy"


def test_callable_policy_inference():
    """Test callable policy uses a function to generate code."""

    def generator(prompt: str) -> str:
        return f"# Generated for: {prompt}\naction = [0.0, 0.0, 0.0]"

    model = CallablePolicy(generator_fn=generator)
    result = model.inference(prompt="lift cube")

    assert "Generated for: lift cube" in result.code
    assert "action = [0.0, 0.0, 0.0]" in result.code
    assert result.policy_name == "CallablePolicy"


def test_policy_base_cannot_instantiate():
    """Test that base PolicyBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        PolicyBase()


def test_core_registries_include_common_components():
    """Test that common core implementations are registered by type."""
    assert BaseAgent.agent_type() == "base_agent"
    assert PolicyBase.get_registered_type("static") is StaticPolicy
    assert PolicyBase.get_registered_type("callable") is CallablePolicy
    assert PolicyBase.get_registered_type("vllm") is VLLMPolicy
    assert PolicyBase.get_registered_type("sam3") is SAM3Policy
    assert PolicyBase.get_registered_type("graspnet") is GraspNetPolicy
    assert PolicyBase.get_registered_type("pyroki") is PyrokiPolicy


def test_agent_register_decorator():
    """Test that agent subclasses can be registered by type."""

    @BaseAgent.register()
    class RegisteredAgent(BaseAgent):
        name = "Registered Agent"

        @classmethod
        def agent_type(cls) -> str:
            return "registered_agent"

        def functions(self):
            return {}

    assert BaseAgent.get_registered_type("registered_agent") is RegisteredAgent
