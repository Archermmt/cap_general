"""Genesis robot controllers."""

from cap_general.frameworks.genesis.robot.drone_robot import DroneHoverRobot, DroneHoverRobotConfig
from cap_general.frameworks.genesis.robot.franka_robot import FrankaRobot, FrankaRobotConfig, ObjConfig
from cap_general.frameworks.genesis.robot.go2_robot import Go2Robot, Go2RobotConfig
from cap_general.frameworks.genesis.robot.grasp_robot import GraspRobot, GraspRobotConfig

__all__ = [
    "DroneHoverRobot",
    "DroneHoverRobotConfig",
    "FrankaRobot",
    "FrankaRobotConfig",
    "ObjConfig",
    "Go2Robot",
    "Go2RobotConfig",
    "GraspRobot",
    "GraspRobotConfig",
]
