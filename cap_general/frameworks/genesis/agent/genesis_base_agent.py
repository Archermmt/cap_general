"""Shared agent helpers for Genesis environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cap_general.core.agent import BaseAgent, BaseAgentConfig


@dataclass
class GenesisBaseAgentConfig(BaseAgentConfig):
    """Shared configuration for Genesis agents."""


class GenesisBaseAgent(BaseAgent):
    """Base class for agents backed by a Genesis robot environment."""

    agent_type = "genesis_base"
    config_cls = GenesisBaseAgentConfig

    def init_genesis(self, gs_scene: Any) -> None:
        self._robot.init_genesis(gs_scene)
