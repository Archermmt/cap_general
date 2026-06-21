"""Robosuite robot controllers."""

from cap_general.frameworks.robosuite.robot.robosuite_base_robot import RobosuiteBaseRobot, RobosuiteBaseRobotConfig
from cap_general.frameworks.robosuite.robot.robosuite_cube_robot import (
    RobosuiteCubeRobot,
    RobosuiteCubeRobotConfig,
)

__all__ = [
    "RobosuiteBaseRobot",
    "RobosuiteBaseRobotConfig",
    "RobosuiteCubeRobot",
    "RobosuiteCubeRobotConfig",
]
