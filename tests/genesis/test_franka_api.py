"""Tests for Genesis Franka env and cube agent."""

import pytest
from cap_general.core.agent import AgentBase
from cap_general.core.env import EnvBase
from cap_general.genesis import FrankaCudaAgent, FrankaEnv


def test_franka_env_method_forwarding():
    """Test that FrankaEnv forwards method calls to the Genesis robot instance."""

    class MockRobot:
        def set_joint_positions(self, positions):
            self.last_positions = positions
            return True

        def get_joint_positions(self):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mock_robot = MockRobot()
    env = FrankaEnv(robot=mock_robot)

    result = env.set_joint_positions([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert result is True
    assert mock_robot.last_positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    positions = env.get_joint_positions()
    assert len(positions) == 7


def test_franka_env_registry_and_base():
    """Test that FrankaEnv is a registered EnvBase subclass."""
    assert issubclass(FrankaEnv, EnvBase)
    assert EnvBase.get_registered_type("genesis_franka") is FrankaEnv


def test_franka_cuda_agent_prompt_docs():
    """Test that combined_doc includes cube-task methods."""
    agent = FrankaCudaAgent(env=FrankaEnv())
    doc = agent.combined_doc()

    assert "get_observation" in doc
    assert "compute_reward" in doc
    assert "run" in doc


def test_franka_cuda_agent_registry_and_base():
    """Test that FrankaCudaAgent is a registered AgentBase subclass."""
    assert issubclass(FrankaCudaAgent, AgentBase)
    assert AgentBase.get_registered_type("genesis_franka_cube") is FrankaCudaAgent


def test_franka_env_with_none_robot():
    """Test that FrankaEnv can be instantiated without a robot."""
    env = FrankaEnv(robot=None)

    assert env.get_joint_positions() == [0.0] * 7
    assert env.grasp() is True
