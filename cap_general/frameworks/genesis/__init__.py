"""Genesis-specific CAP components."""

from __future__ import annotations

import importlib

# Trigger registration of all Genesis components
import cap_general.frameworks.genesis.agent  # noqa: F401
import cap_general.frameworks.genesis.policy  # noqa: F401
import cap_general.frameworks.genesis.robot  # noqa: F401
import cap_general.frameworks.genesis.scene  # noqa: F401
import cap_general.frameworks.genesis.pipeline.job  # noqa: F401

__all__ = [
    "GenesisBaseAgent",
    "GenesisDroneAgent",
    "GenesisFrankaAgent",
    "GenesisGo2Agent",
    "GenesisGraspAgent",
    "GenesisDroneRobot",
    "GenesisFrankaRobot",
    "GenesisGo2Robot",
    "GenesisGraspRobot",
    "ObjConfig",
    "GenesisScene",
    "BehaviorCloningPolicy",
    "GenesisTrainJob",
]

_LAZY: dict[str, tuple[str, str]] = {
    "GenesisBaseAgent": ("cap_general.frameworks.genesis.agent", "GenesisBaseAgent"),
    "GenesisDroneAgent": ("cap_general.frameworks.genesis.agent", "GenesisDroneAgent"),
    "GenesisFrankaAgent": ("cap_general.frameworks.genesis.agent", "GenesisFrankaAgent"),
    "GenesisGo2Agent": ("cap_general.frameworks.genesis.agent", "GenesisGo2Agent"),
    "GenesisGraspAgent": ("cap_general.frameworks.genesis.agent", "GenesisGraspAgent"),
    "GenesisDroneRobot": ("cap_general.frameworks.genesis.robot", "GenesisDroneRobot"),
    "GenesisFrankaRobot": ("cap_general.frameworks.genesis.robot", "GenesisFrankaRobot"),
    "GenesisGo2Robot": ("cap_general.frameworks.genesis.robot", "GenesisGo2Robot"),
    "GenesisGraspRobot": ("cap_general.frameworks.genesis.robot", "GenesisGraspRobot"),
    "ObjConfig": ("cap_general.frameworks.genesis.robot", "ObjConfig"),
    "GenesisScene": ("cap_general.frameworks.genesis.scene", "GenesisScene"),
    "BehaviorCloningPolicy": ("cap_general.frameworks.genesis.policy", "BehaviorCloningPolicy"),
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
