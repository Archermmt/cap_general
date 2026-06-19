"""Utilities for loading Genesis example modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def load_module_from_file(module_name: str, file_path: str | Path) -> ModuleType:
    """Load a Python module from an example file path."""
    path = Path(file_path).expanduser().resolve()
    module_dir = str(path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
