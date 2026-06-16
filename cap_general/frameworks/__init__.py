"""Framework-specific CAP components."""

from __future__ import annotations

import importlib
import pkgutil


def import_frameworks() -> None:
    """Import all bundled framework packages so their registries are populated."""
    package_prefix = f"{__name__}."
    for module_info in pkgutil.iter_modules(__path__, package_prefix):
        if module_info.ispkg:
            importlib.import_module(module_info.name)


__all__ = ["import_frameworks"]
