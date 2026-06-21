"""Tests for core CAP agent, robot, and policy primitives."""

import pytest

from cap_general.core.agent import BaseAgent, BaseAgentConfig
from cap_general.core.robot import BaseRobot
from cap_general.core.policy import (
    BasePolicy,
    GraspNetPolicy,
    PyrokiPolicy,
    SAM3Policy,
)


@BaseRobot.register()
class CoreDummyRobot(BaseRobot):
    """Small robot for core agent tests."""

    name = "Core Dummy Robot"

    @classmethod
    def robot_type(cls) -> str:
        return "core_dummy"

    def _reset(self, options=None):
        return {"step": self.step_cnt}, {"seed": self._seed, "options": options or {}}

    def _step(self, action):
        return {"step": self.step_cnt}, 0.0, False, False, {"action": action}

    def get_observation(self, folder):
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


def test_agent__function_doc():
    """Test that _function_doc extracts method signatures and docstrings."""
    agent = SimpleAgent(config=BaseAgentConfig(robot={"type": "core_dummy"}))
    doc = agent._function_doc()

    assert "add" in doc
    assert "multiply" in doc
    assert "Add two numbers" in doc
    assert "Multiply two numbers" in doc
    assert "a: int" in doc or "a:int" in doc
    assert "b: int" in doc or "b:int" in doc


def test_agent_doc_reset_options_use_reset_rules():
    """Test that agent_doc embeds reset rules in reset documentation."""
    agent = SimpleAgent(config=BaseAgentConfig(robot={"type": "core_dummy"}))
    doc = agent.agent_doc()

    assert "reset_rules" in doc
    assert "reset_doc" in doc
    assert doc["reset_rules"] in doc["reset_doc"]
    assert 'agent_doc()["reset_rules"]' not in doc["reset_doc"]


def test_policy_base_cannot_instantiate():
    """Test that base BasePolicy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BasePolicy()


def test_core_registries_include_common_components():
    """Test that common core implementations are registered by type."""
    assert BaseAgent.agent_type() == "base_agent"
    assert BasePolicy.get_registered_class("sam3") is SAM3Policy
    assert BasePolicy.get_registered_class("graspnet") is GraspNetPolicy
    assert BasePolicy.get_registered_class("pyroki") is PyrokiPolicy


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

    assert BaseAgent.get_registered_class("registered_agent") is RegisteredAgent
