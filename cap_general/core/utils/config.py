"""Configuration loading and command-line override helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


def parse_cli_overrides(args: list[str]) -> list[str]:
    """Convert ``--nested.path value`` arguments to OmegaConf dotlist entries."""
    overrides: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--") or token == "--":
            raise ValueError(f"Expected a configuration option starting with '--', got {token!r}")

        option = token[2:]
        if "=" in option:
            key, value = option.split("=", 1)
        else:
            key = option
            index += 1
            if index >= len(args) or args[index].startswith("--"):
                raise ValueError(f"Missing value for configuration option --{key}")
            value = args[index]

        if not key:
            raise ValueError("Configuration override key cannot be empty")
        overrides.append(f"{key}={value}")
        index += 1
    return overrides


def load_yaml_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML, apply strict recursive overrides, and return plain data."""
    config = OmegaConf.load(Path(config_path))
    if not OmegaConf.is_dict(config):
        raise TypeError("Scene yaml config must contain a mapping at the top level")
    if "type" not in config:
        config.type = "scene"

    OmegaConf.set_struct(config, True)
    for override in overrides or []:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ValueError(f"Invalid configuration override: {override!r}")
        parsed_value = OmegaConf.from_dotlist([f"value={raw_value}"]).value
        OmegaConf.update(config, key, parsed_value, merge=True)

    data = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(data, dict):
        raise TypeError("Scene yaml config must contain a mapping at the top level")
    return data
