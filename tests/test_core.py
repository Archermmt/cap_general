"""Tests for core CAP agent, robot, and policy primitives."""

import logging

import pytest

from cap_general.core.agent import BaseAgent, BaseAgentConfig
from cap_general.core.operator import BaseOperator
from cap_general.core.robot import BaseRobot
from cap_general.core.policy import BasePolicy

LOGGER = logging.getLogger(__name__)


@BaseRobot.register()
class CoreDummyRobot(BaseRobot):
    """Small robot for core agent tests."""

    robot_type = "core_dummy"
    config_cls = BaseRobot.config_cls

    def _reset(self, options=None):
        return {"step": self.step_cnt}, {"seed": self._config.seed, "options": options or {}}

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
    agent = SimpleAgent(config=BaseAgentConfig(robot={"type": "core_dummy"}), logger=LOGGER)
    doc = agent._function_doc()

    assert "add" in doc
    assert "multiply" in doc
    assert "Add two numbers" in doc
    assert "Multiply two numbers" in doc
    assert "a: int" in doc or "a:int" in doc
    assert "b: int" in doc or "b:int" in doc


def test_agent_doc_embeds_reset_options_in_function_doc():
    """Test that agent_doc exposes reset options through function_doc."""
    agent = SimpleAgent(config=BaseAgentConfig(robot={"type": "core_dummy"}), logger=LOGGER)
    doc = agent.agent_doc()

    assert "function_doc" in doc
    assert "reset_level: 0 resets only the robot pose" in doc["function_doc"]
    assert '[_options_doc()]' not in doc["function_doc"]


def test_policy_base_cannot_instantiate():
    """Test that base BasePolicy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BasePolicy()


def test_core_registries_include_common_components():
    """Test that common operators are registered in the operator registry."""
    assert BaseAgent.agent_type == "base"
    assert BaseOperator.get_registered_class("model", "sam3") is not None
    assert BaseOperator.get_registered_class("model", "graspnet") is not None
    assert BaseOperator.get_registered_class("model", "pyroki") is not None
    assert BaseOperator.get_registered_class("model", "rsl_rl") is not None


def test_agent_register_decorator():
    """Test that agent subclasses can be registered by type."""

    @BaseAgent.register()
    class RegisteredAgent(BaseAgent):
        agent_type = "registered"
        config_cls = BaseAgentConfig

        def functions(self):
            return {}

    assert BaseAgent.get_registered_class("registered") is RegisteredAgent
