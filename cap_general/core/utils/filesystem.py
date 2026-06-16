"""Filesystem helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from cap_general.core.utils.serialization import to_json_safe


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
