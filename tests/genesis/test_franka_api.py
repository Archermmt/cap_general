"""Tests for Genesis Franka API."""

import pytest
from cap_general.genesis.apis.franka import GenesisFrankaApi


def test_franka_api_method_forwarding():
    """Test that GenesisFrankaApi forwards method calls to the robot instance."""

    # Create a mock robot object
    class MockRobot:
        def set_joint_positions(self, positions):
            self.last_positions = positions
            return True

        def get_joint_positions(self):
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mock_robot = MockRobot()
    api = GenesisFrankaApi(robot=mock_robot)

    # Test method forwarding
    result = api.set_joint_positions([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert result is True
    assert mock_robot.last_positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    positions = api.get_joint_positions()
    assert len(positions) == 7


def test_franka_api_prompt_docs():
    """Test that combined_doc includes Franka control methods."""

    class MockRobot:
        def set_joint_positions(self, positions):
            """Set joint positions."""
            pass

        def get_joint_positions(self):
            """Get current joint positions."""
            pass

        def set_gripper_position(self, width):
            """Set gripper width."""
            pass

    mock_robot = MockRobot()
    api = GenesisFrankaApi(robot=mock_robot)
    doc = api.combined_doc()

    assert "set_joint_positions" in doc
    assert "get_joint_positions" in doc
    assert "set_gripper_position" in doc


def test_franka_api_with_none_robot():
    """Test that API can be instantiated without a robot (for testing)."""
    api = GenesisFrankaApi(robot=None)

    # Should still have the base methods
    assert hasattr(api, "combined_doc")
    assert hasattr(api, "api_spec")
