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


@pytest.mark.parametrize(
    "checkpoint_config",
    [
        {"ckpt_dir": None},
        {"ckpt_dir": "/path/that/does/not/exist"},
        {"ckpt_dir": None, "ckpt_path": "/path/that/does/not/exist/model.pt"},
    ],
)
def test_rsl_rl_op_randomly_initializes_once_when_checkpoint_is_unavailable(checkpoint_config, caplog):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    TensorDict = pytest.importorskip("tensordict").TensorDict
    from cap_general.core.operator.model.rsl_rl_op import RslRlOp

    config = {
        **checkpoint_config,
        "obs_dim": 5,
        "action_dim": 2,
        "actor_cfg": {
            "class_name": "MLPModel",
            "hidden_dims": [8],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
    }
    operator = RslRlOp(config=config, logger=LOGGER)
    with caplog.at_level(logging.WARNING):
        operator.reset()
        warning_count = len(caplog.records)
        obs = TensorDict({"policy": torch.zeros((3, 5))}, batch_size=[3])
        first = operator.inference({"obs": obs})
        second = operator.inference({"obs": obs})

    assert operator.get_model() is not None
    assert first.shape == (3, 2)
    assert second.shape == (3, 2)
    assert warning_count == 1
    assert len(caplog.records) == warning_count
    assert "randomly initialized actor weights" in caplog.records[0].message


def test_agent_register_decorator():
    """Test that agent subclasses can be registered by type."""

    @BaseControl.register()
    class RegisteredAgent(BaseControl):
        control_type = "registered"
        config_cls = BaseControlConfig

        def functions(self):
            return {}

    assert BaseControl.get_registered_class("registered") is RegisteredAgent
