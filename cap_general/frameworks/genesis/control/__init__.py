"""Genesis task controls."""

from cap_general.frameworks.genesis.control.genesis_base_control import GenesisBaseControl, GenesisBaseControlConfig
from cap_general.frameworks.genesis.control.genesis_drone_control import GenesisDroneControl, GenesisDroneControlConfig
from cap_general.frameworks.genesis.control.genesis_go2_control import GenesisGo2Control, GenesisGo2ControlConfig
from cap_general.frameworks.genesis.control.genesis_grasp_control import GenesisGraspControl, GenesisGraspControlConfig
from cap_general.frameworks.genesis.control.genesis_humanoid_control import (
    GenesisHumanoidControl,
    GenesisHumanoidControlConfig,
)

__all__ = [
    "GenesisDroneControl",
    "GenesisDroneControlConfig",
    "GenesisBaseControl",
    "GenesisBaseControlConfig",
    "GenesisGo2Control",
    "GenesisGo2ControlConfig",
    "GenesisGraspControl",
    "GenesisGraspControlConfig",
    "GenesisHumanoidControl",
    "GenesisHumanoidControlConfig",
]
