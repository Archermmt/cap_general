"""Utilities for loading Genesis example modules."""

from __future__ import annotations

import importlib.util
import sys
import threading
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


def step_scene(scene) -> None:
    """Step a Genesis scene through its CAP thread-aware dispatcher when present.

    On macOS Genesis runs the interactive viewer in the main thread. CAP agent
    code can run in a worker thread, so GenesisScene installs ``_cap_step_scene``
    on the raw scene to send step requests back to the viewer-owning thread.
    """
    stepper = getattr(scene, "_cap_step_scene", None)
    if callable(stepper):
        stepper()
        return
    if threading.current_thread() is threading.main_thread():
        scene.step()
    else:
        scene.step(refresh_visualizer=False)
