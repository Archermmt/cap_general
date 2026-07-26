"""Shared control helpers for Genesis environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cap_general.core.control import BaseControl, BaseControlConfig


@dataclass
class GenesisBaseControlConfig(BaseControlConfig):
    """Shared configuration for Genesis controls."""


class GenesisBaseControl(BaseControl):
    """Base class for controls backed by a Genesis robot environment."""

    control_type = "genesis_base"
    config_cls = GenesisBaseControlConfig

    def init_genesis(self, gs_scene: Any) -> None:
        self._robot.init_genesis(gs_scene)
