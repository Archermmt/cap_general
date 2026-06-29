"""Filesystem helpers."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from cap_general.core.utils.serialization import to_json_safe


def load_module_from_file(module_name: str, file_path: str | Path) -> ModuleType:
    """Load a Python module from a file path."""
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


def remove_path(path: str | Path) -> None:
    """Remove a file, symlink, or directory tree if it exists."""
    resolved_path = Path(path)
    if not resolved_path.exists() and not resolved_path.is_symlink():
        return
    if resolved_path.is_dir() and not resolved_path.is_symlink():
        shutil.rmtree(resolved_path)
    else:
        resolved_path.unlink()


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with UTF-8 encoding and stable indentation."""
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(to_json_safe(data), file, ensure_ascii=False, indent=2, default=str)


def write_text(path: str | Path, text: str) -> None:
    """Write text with UTF-8 encoding, ensuring a trailing newline."""
    with Path(path).open("w", encoding="utf-8") as file:
        file.write(text)
        if text and not text.endswith("\n"):
            file.write("\n")
