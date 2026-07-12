"""Compatibility wrapper for the old base_agent module path."""

from cap_general.core.control.base_control import BaseControl, BaseControlConfig, Tee

BaseAgent = BaseControl
BaseAgentConfig = BaseControlConfig

__all__ = ["BaseAgent", "BaseAgentConfig", "BaseControl", "BaseControlConfig", "Tee"]
