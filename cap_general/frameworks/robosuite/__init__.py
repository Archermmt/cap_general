"""Robosuite-specific CAP components."""

from cap_general.frameworks.robosuite.control import (
    PROMPT,
    RobosuiteControl,
    RobosuiteControlConfig,
)
from cap_general.frameworks.robosuite.robot import (
    RobosuiteBaseRobot,
    RobosuiteBaseRobotConfig,
    RobosuiteCubeRobot,
    RobosuiteCubeRobotConfig,
)

__all__ = [
    "RobosuiteBaseRobot",
    "RobosuiteBaseRobotConfig",
    "RobosuiteCubeRobot",
    "RobosuiteCubeRobotConfig",
    "RobosuiteControl",
    "RobosuiteControlConfig",
    "PROMPT",
]
