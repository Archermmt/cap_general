"""Genesis Franka environment controller."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, SupportsFloat

from cap_general.core.env import BaseEnv, BaseEnvConfig


@dataclass
class FrankaEnvConfig(BaseEnvConfig):
    """Configuration for FrankaEnv."""

    robot: Any | None = None


@BaseEnv.register()
class FrankaEnv(BaseEnv):
    """Basic Franka environment operations for Genesis."""

    name = "Genesis Franka Env"
    config_cls = FrankaEnvConfig

    def __init__(
        self,
        config: FrankaEnvConfig,
        logger: logging.Logger | None = None,
    ):
        """Initialize with an optional Genesis robot instance."""
        super().__init__(config=config, logger=logger)
        self._robot = config.robot

    @classmethod
    def env_type(cls) -> str:
        return "genesis_franka"

    @property
    def robot(self) -> Any | None:
        """Return the wrapped Genesis robot instance."""
        return self._robot

    def attach(self, robot: Any):
        """Attach a Genesis robot instance."""
        self._robot = robot

    def set_joint_positions(self, positions: List[float]) -> bool:
        """Set the robot's joint positions."""
        if self._robot is None:
            print(f"[Mock] set_joint_positions called with {positions}")
            return True

        try:
            if hasattr(self._robot, "set_joint_positions"):
                self._robot.set_joint_positions(positions)
            elif hasattr(self._robot, "set_qpos"):
                self._robot.set_qpos(positions)
            else:
                raise AttributeError("Robot does not support setting joint positions")
            return True
        except Exception as e:
            print(f"Error setting joint positions: {e}")
            return False

    def get_joint_positions(self) -> List[float]:
        """Get the current joint positions."""
        if self._robot is None:
            print("[Mock] get_joint_positions called")
            return [0.0] * 7

        try:
            if hasattr(self._robot, "get_joint_positions"):
                return list(self._robot.get_joint_positions())
            if hasattr(self._robot, "get_qpos"):
                return self._robot.get_qpos().tolist()
            raise AttributeError("Robot does not support getting joint positions")
        except Exception as e:
            print(f"Error getting joint positions: {e}")
            return [0.0] * 7

    def set_gripper_position(self, width: float) -> bool:
        """Set the gripper opening width."""
        if self._robot is None:
            print(f"[Mock] set_gripper_position called with width={width}")
            return True

        try:
            if hasattr(self._robot, "set_gripper_position"):
                self._robot.set_gripper_position(width)
            else:
                raise AttributeError("Robot does not support set_gripper_position")
            return True
        except Exception as e:
            print(f"Error setting gripper position: {e}")
            return False

    def get_ee_pose(self) -> List[float]:
        """Get the end-effector pose as [x, y, z, qx, qy, qz, qw]."""
        if self._robot is None:
            print("[Mock] get_ee_pose called")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        try:
            if hasattr(self._robot, "get_ee_pose"):
                return list(self._robot.get_ee_pose())
            raise AttributeError("Robot does not support get_ee_pose")
        except Exception as e:
            print(f"Error getting EE pose: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    def move_to_pose(self, x: float, y: float, z: float, duration: float = 1.0) -> bool:
        """Move the end-effector to a Cartesian position."""
        if self._robot is None:
            print(f"[Mock] move_to_pose called with x={x}, y={y}, z={z}, duration={duration}")
            return True

        try:
            if hasattr(self._robot, "move_to_pose"):
                self._robot.move_to_pose(x, y, z, duration)
            else:
                raise AttributeError("Robot does not support move_to_pose")
            return True
        except Exception as e:
            print(f"Error moving to pose: {e}")
            return False

    def grasp(self) -> bool:
        """Close the gripper to grasp an object."""
        return self.set_gripper_position(0.0)

    def release(self) -> bool:
        """Open the gripper to release an object."""
        return self.set_gripper_position(0.08)

    def _reset(
        self,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset environment state if the wrapped robot supports it."""
        if self._robot is not None and hasattr(self._robot, "reset"):
            self._robot.reset()
        return self.get_observation(folder=Path(".")), {"seed": self._seed, "options": options or {}}

    def _step(
        self, action: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        """Apply one low-level action and return a Gymnasium step tuple."""
        action = action or {}
        if "joint_positions" in action:
            self.set_joint_positions(action["joint_positions"])
        if "gripper_width" in action:
            self.set_gripper_position(float(action["gripper_width"]))
        if "pose" in action:
            pose = action["pose"]
            self.move_to_pose(float(pose[0]), float(pose[1]), float(pose[2]))

        obs = self.get_observation(folder=Path("."))
        return obs, self._compute_reward(), self.task_completed(), False, {"action": action}

    def get_observation(self, folder: str | Path) -> dict[str, Any]:
        """Return a lightweight environment observation."""
        return {
            "joint_positions": self.get_joint_positions(),
            "ee_pose": self.get_ee_pose(),
        }

    def _compute_reward(self) -> SupportsFloat:
        """Compute the current reward."""
        return 0.0

    def task_completed(self) -> bool:
        """Return whether the low-level robot task is complete."""
        return False
