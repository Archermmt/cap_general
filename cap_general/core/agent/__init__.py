"""Compatibility exports for the old agent module path."""

from cap_general.core.control import BaseControl, BaseControlConfig

BaseAgent = BaseControl
BaseAgentConfig = BaseControlConfig

__all__ = ["BaseAgent", "BaseAgentConfig", "BaseControl", "BaseControlConfig"]
