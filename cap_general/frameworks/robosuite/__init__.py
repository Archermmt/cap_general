"""Robosuite-specific CAP components."""

from cap_general.frameworks.robosuite.agent import (
    PROMPT,
    RobosuiteAgent,
    RobosuiteAgentConfig,
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
    "RobosuiteAgent",
    "RobosuiteAgentConfig",
    "PROMPT",
]
