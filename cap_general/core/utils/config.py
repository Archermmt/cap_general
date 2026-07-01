"""Configuration loading and command-line override helpers."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

from omegaconf import OmegaConf


def build_dataclass_config(config_cls, config_data: dict[str, Any]):
    """Build a dataclass config from a dictionary."""
    config_fields = fields(config_cls)
    field_names = {field.name for field in config_fields}
    if "reset_mode" in field_names and "reset_frequency" not in field_names:
        config_data = dict(config_data)
        legacy_reset_frequency = config_data.pop("reset_frequency", None)
        if legacy_reset_frequency is not None:
            config_data.setdefault("reset_mode", legacy_reset_frequency)
    type_hints = get_type_hints(config_cls)
    values = {
        key: coerce_dataclass_field(type_hints.get(key), value)
        for key, value in config_data.items()
        if key in field_names
    }
    return config_cls(**values)


def coerce_dataclass_field(field_type: Any, value: Any) -> Any:
    """Build nested dataclass fields from dictionaries when the annotation is concrete."""
    if isinstance(value, dict) and "type" in value:
        return value
    if isinstance(value, dict) and isinstance(field_type, type) and is_dataclass(field_type):
        return build_dataclass_config(field_type, value)
    return value


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
