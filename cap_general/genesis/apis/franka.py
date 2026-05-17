"""Genesis Franka robot API for CAP."""

from typing import List, Optional
from cap_general.core.apis.base import CapApiBase


class GenesisFrankaApi(CapApiBase):
    """API for controlling a Franka Emika Panda robot in Genesis.

    Provides high-level control methods for robot manipulation tasks.
    Wraps a Genesis robot instance and exposes it to the policy model.
    """

    def __init__(self, robot=None):
        """Initialize the Franka API.

        Args:
            robot: Genesis robot instance. If None, methods will raise errors.
        """
        self._robot = robot

    def set_joint_positions(self, positions: List[float]) -> bool:
        """Set the robot's joint positions.

        Args:
            positions: List of 7 joint angles in radians.

        Returns:
            True if successful, False otherwise.
        """
        if self._robot is None:
            print(f"[Mock] set_joint_positions called with {positions}")
            return True

        try:
            # Forward to Genesis robot
            if hasattr(self._robot, "set_joint_positions"):
                self._robot.set_joint_positions(positions)
                return True
            else:
                raise AttributeError("Robot does not support set_joint_positions")
        except Exception as e:
            print(f"Error setting joint positions: {e}")
            return False

    def get_joint_positions(self) -> List[float]:
        """Get the current joint positions.

        Returns:
            List of 7 joint angles in radians.
        """
        if self._robot is None:
            print("[Mock] get_joint_positions called")
            return [0.0] * 7

        try:
            if hasattr(self._robot, "get_joint_positions"):
                return self._robot.get_joint_positions()
            else:
                raise AttributeError("Robot does not support get_joint_positions")
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return [0.0] * 7

    def set_gripper_position(self, width: float) -> bool:
        """Set the gripper opening width.

        Args:
            width: Gripper width in meters (0.0 = closed, ~0.08 = fully open).

        Returns:
            True if successful, False otherwise.
        """
        if self._robot is None:
            print(f"[Mock] set_gripper_position called with width={width}")
            return True

        try:
            if hasattr(self._robot, "set_gripper_position"):
                self._robot.set_gripper_position(width)
                return True
            else:
                raise AttributeError("Robot does not support set_gripper_position")
        except Exception as e:
            print(f"Error setting gripper position: {e}")
            return False

    def get_ee_pose(self) -> List[float]:
        """Get the end-effector pose.

        Returns:
            List of 7 values [x, y, z, qx, qy, qz, qw] representing position and orientation.
        """
        if self._robot is None:
            print("[Mock] get_ee_pose called")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        try:
            if hasattr(self._robot, "get_ee_pose"):
                return self._robot.get_ee_pose()
            else:
                raise AttributeError("Robot does not support get_ee_pose")
        except Exception as e:
            print(f"Error getting EE pose: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def move_to_pose(self, x: float, y: float, z: float, duration: float = 1.0) -> bool:
        """Move the end-effector to a Cartesian pose.

        Args:
            x: Target x position in meters.
            y: Target y position in meters.
            z: Target z position in meters.
            duration: Movement duration in seconds.

        Returns:
            True if successful, False otherwise.
        """
        if self._robot is None:
            print(
                f"[Mock] move_to_pose called with x={x}, y={y}, z={z}, duration={duration}"
            )
            return True

        try:
            if hasattr(self._robot, "move_to_pose"):
                self._robot.move_to_pose(x, y, z, duration)
                return True
            else:
                raise AttributeError("Robot does not support move_to_pose")
        except Exception as e:
            print(f"Error moving to pose: {e}")
            return False

    def grasp(self) -> bool:
        """Close the gripper to grasp an object.

        Returns:
            True if successful, False otherwise.
        """
        return self.set_gripper_position(0.0)

    def release(self) -> bool:
        """Open the gripper to release an object.

        Returns:
            True if successful, False otherwise.
        """
        return self.set_gripper_position(0.08)
