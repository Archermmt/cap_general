"""Tests for core CAP agent, robot, and policy primitives."""

import logging

import pytest

from cap_general.core.control import BaseControl, BaseControlConfig
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


class SimpleAgent(BaseControl):
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
    agent = SimpleAgent(config=BaseControlConfig(robot={"type": "core_dummy"}), logger=LOGGER)
    doc = agent._function_doc()

    assert "add" in doc
    assert "multiply" in doc
    assert "Add two numbers" in doc
    assert "Multiply two numbers" in doc
    assert "a: int" in doc or "a:int" in doc
    assert "b: int" in doc or "b:int" in doc


def test_control_doc_embeds_reset_options_in_function_doc():
    """Test that control_doc exposes reset options through function_doc."""
    agent = SimpleAgent(config=BaseControlConfig(robot={"type": "core_dummy"}), logger=LOGGER)
    doc = agent.control_doc()

    assert "function_doc" in doc
    assert "reset_level: 0 resets only the robot pose" in doc["function_doc"]
    assert '[_options_doc()]' not in doc["function_doc"]


def test_agent_default_reward_uses_robot_last_reward():
    agent = SimpleAgent(config=BaseControlConfig(robot={"type": "core_dummy"}), logger=LOGGER)
    agent._robot._last_reward = 2.5

    assert agent._compute_reward() == 2.5


def test_agent_records_each_execute_by_default():
    agent = SimpleAgent(config=BaseControlConfig(robot={"type": "core_dummy", "reset_time": 0}), logger=LOGGER)
    record_calls = []

    def fake_record(step_idx: int = -1, clean_frames: bool = False):
        record_calls.append({"step_idx": step_idx, "clean_frames": clean_frames})
        return {}

    agent.record = fake_record

    result = agent.execute("RESULT = add(1, 2)")

    assert result["ok"] is True
    assert result["result"] == 3
    assert record_calls == [{"step_idx": 0, "clean_frames": False}]


def test_agent_can_disable_execute_recording():
    agent = SimpleAgent(
        config=BaseControlConfig(
            robot={"type": "core_dummy", "reset_time": 0},
            record_execute=False,
        ),
        logger=LOGGER,
    )
    record_calls = []

    def fake_record(step_idx: int = -1, clean_frames: bool = False):
        record_calls.append({"step_idx": step_idx, "clean_frames": clean_frames})
        return {}

    agent.record = fake_record

    result = agent.execute("RESULT = multiply(2, 4)")

    assert result["ok"] is True
    assert result["result"] == 8
    assert record_calls == []


def test_execute_result_omits_agent_judgment_fields_from_observation():
    agent = SimpleAgent(
        config=BaseControlConfig(robot={"type": "core_dummy", "reset_time": 0}),
        logger=LOGGER,
    )
    agent._robot.get_observation = lambda _folder: {
        "position": [1.0, 2.0, 3.0],
        "reward": 0.9,
        "done": True,
        "mock": False,
    }

    result = agent.execute("RESULT = True")

    assert "reward" not in result
    assert result["obs"] == {"position": [1.0, 2.0, 3.0]}
    assert agent.get_obs() == {
        "position": [1.0, 2.0, 3.0],
        "reward": 0.9,
        "done": True,
        "mock": False,
    }


def test_policy_base_cannot_instantiate():
    """Test that base BasePolicy cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BasePolicy()


def test_core_registries_include_common_components():
    """Test that common operators are registered in the operator registry."""
    assert BaseControl.control_type == "base"
    assert BaseOperator.get_registered_class("model", "sam3") is not None
    assert BaseOperator.get_registered_class("model", "graspnet") is not None
    assert BaseOperator.get_registered_class("model", "pyroki") is not None
    assert BaseOperator.get_registered_class("model", "rsl_rl") is not None


@pytest.mark.parametrize("ckpt_dir", [None, "/path/that/does/not/exist"])
def test_rsl_rl_op_skips_weights_when_checkpoint_directory_is_unavailable(ckpt_dir):
    from cap_general.core.operator.model.rsl_rl_op import RslRlOp

    operator = RslRlOp(config={"ckpt_dir": ckpt_dir}, logger=LOGGER)
    operator.reset()

    assert operator.get_model() is None


def test_agent_register_decorator():
    """Test that agent subclasses can be registered by type."""

    @BaseControl.register()
    class RegisteredAgent(BaseControl):
        control_type = "registered"
        config_cls = BaseControlConfig

        def functions(self):
            return {}

    assert BaseControl.get_registered_class("registered") is RegisteredAgent
