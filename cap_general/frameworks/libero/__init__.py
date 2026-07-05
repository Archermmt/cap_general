"""LIBERO-specific CAP components."""

from __future__ import annotations

import importlib

# Trigger registration of all LIBERO components
import cap_general.frameworks.libero.agent  # noqa: F401
import cap_general.frameworks.libero.robot  # noqa: F401
import cap_general.frameworks.libero.pipeline.job  # noqa: F401

__all__ = [
    "LiberoAgent",
    "LiberoRobot",
    "LiberoTrainJob",
]

_LAZY: dict[str, tuple[str, str]] = {
    "LiberoAgent": ("cap_general.frameworks.libero.agent", "LiberoAgent"),
    "LiberoRobot": ("cap_general.frameworks.libero.robot", "LiberoRobot"),
    "LiberoTrainJob": ("cap_general.frameworks.libero.pipeline.job", "LiberoTrainJob"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY:
        module_path, attr_name = _LAZY[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
