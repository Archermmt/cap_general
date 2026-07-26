"""Genesis-specific CAP components."""

from __future__ import annotations

import importlib

# Trigger registration of all Genesis components
import cap_general.frameworks.genesis.control  # noqa: F401
import cap_general.frameworks.genesis.policy  # noqa: F401
import cap_general.frameworks.genesis.robot  # noqa: F401
import cap_general.frameworks.genesis.scene  # noqa: F401
import cap_general.frameworks.genesis.pipeline.job  # noqa: F401

__all__ = [
    "GenesisBaseControl",
    "GenesisDroneControl",
    "GenesisGo2Control",
    "GenesisGraspControl",
    "GenesisHumanoidControl",
    "GenesisDroneRobot",
    "GenesisFrankaRobot",
    "GenesisGo2Robot",
    "GenesisGraspRobot",
    "GenesisHumanoidRobot",
    "ObjConfig",
    "GenesisScene",
    "BehaviorCloningPolicy",
    "GenesisEvalJob",
    "GenesisTrainJob",
]

_LAZY: dict[str, tuple[str, str]] = {
    "GenesisBaseControl": ("cap_general.frameworks.genesis.control", "GenesisBaseControl"),
    "GenesisDroneControl": ("cap_general.frameworks.genesis.control", "GenesisDroneControl"),
    "GenesisGo2Control": ("cap_general.frameworks.genesis.control", "GenesisGo2Control"),
    "GenesisGraspControl": ("cap_general.frameworks.genesis.control", "GenesisGraspControl"),
    "GenesisHumanoidControl": ("cap_general.frameworks.genesis.control", "GenesisHumanoidControl"),
    "GenesisDroneRobot": ("cap_general.frameworks.genesis.robot", "GenesisDroneRobot"),
    "GenesisFrankaRobot": ("cap_general.frameworks.genesis.robot", "GenesisFrankaRobot"),
    "GenesisGo2Robot": ("cap_general.frameworks.genesis.robot", "GenesisGo2Robot"),
    "GenesisGraspRobot": ("cap_general.frameworks.genesis.robot", "GenesisGraspRobot"),
    "GenesisHumanoidRobot": ("cap_general.frameworks.genesis.robot", "GenesisHumanoidRobot"),
    "ObjConfig": ("cap_general.frameworks.genesis.robot", "ObjConfig"),
    "GenesisScene": ("cap_general.frameworks.genesis.scene", "GenesisScene"),
    "BehaviorCloningPolicy": ("cap_general.frameworks.genesis.policy", "BehaviorCloningPolicy"),
    "GenesisEvalJob": ("cap_general.frameworks.genesis.pipeline.job", "GenesisEvalJob"),
    "GenesisTrainJob": ("cap_general.frameworks.genesis.pipeline.job", "GenesisTrainJob"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
