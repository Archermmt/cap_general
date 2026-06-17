"""Tests for RobosuiteAgent using FrankaPickPlaceCodeEnv oracle code."""

from cap_general.core.agent import BaseAgent
from cap_general.core.env import BaseEnv
from cap_general.frameworks.robosuite import (
    ORACLE_CODE,
    MockRobosuiteCubeEnv,
    RobosuiteAgent,
    RobosuiteAgentConfig,
    RobosuiteCubeEnv,
    RobosuiteEnv,
)


def test_robosuite_agent_registry_and_doc():
    """Test that RobosuiteAgent registers and exposes Franka API helpers."""
    assert BaseAgent.get_registered_class("robosuite") is RobosuiteAgent
    assert BaseEnv.get_registered_class("robosuite") is RobosuiteEnv
    assert issubclass(RobosuiteCubeEnv, object)

    agent = RobosuiteAgent(RobosuiteAgentConfig(record_dir="outputs/robosuite_test"))
    doc = agent.agent_doc()

    assert "get_object_pose" in doc["function_doc"]
    assert "sample_grasp_pose" in doc["function_doc"]
    assert "goto_pose" in doc["function_doc"]
    assert isinstance(agent.env.low_level_env, MockRobosuiteCubeEnv)


def test_robosuite_agent_oracle_code_executes():
    """Test Cap-X FrankaPickPlaceCodeEnv oracle code through RobosuiteAgent."""
    agent = RobosuiteAgent(RobosuiteAgentConfig(record_dir="outputs/robosuite_test"))
    agent.reset(options={})

    result = agent.execute(ORACLE_CODE)
    record = agent.record(step_idx=-1)

    assert result["ok"] is True
    assert result["result"]["success"] is True
    assert record["info"]["total_execute"] == 1
    assert "get_object_pose" in record["code"]
