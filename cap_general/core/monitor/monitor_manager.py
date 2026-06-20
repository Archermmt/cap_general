"""Process-global monitor manager."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from cap_general.core.monitor.base_monitor import BaseMonitor, BaseMonitorConfig


@dataclass
class MonitorConfig:
    """Configuration for optional environment-owned monitoring."""

    enable: bool = False
    name: str = "default"
    config: dict[str, Any] | None = None


class MonitorManager:
    """Singleton-style manager for creating and invoking monitors."""

    _instance: "MonitorManager | None" = None

    def __init__(self, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)
        self._monitors: dict[str, BaseMonitor] = {}

    @classmethod
    def instance(cls) -> "MonitorManager":
        """Return the process-global monitor manager."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def create_monitor(
        self,
        name: str = "default",
        monitor_cls: type[BaseMonitor] = BaseMonitor,
        config: MonitorConfig | BaseMonitorConfig | dict[str, Any] | None = None,
    ) -> BaseMonitor | None:
        """Create or return a monitor by name."""
        if self._is_env_monitor_config(config):
            monitor_config = self._build_env_monitor_config(config)
            if not monitor_config.enable:
                return None
            name = monitor_config.name
            config = monitor_config.config

        monitor = self._monitors.get(name)
        if monitor is None:
            config_data = self._build_monitor_from_config_data(name, monitor_cls, config)
            monitor = BaseMonitor.from_config(config_data, logger=self._logger)
            self._monitors[name] = monitor
        return monitor

    @staticmethod
    def _is_env_monitor_config(config: Any) -> bool:
        if isinstance(config, MonitorConfig):
            return True
        return isinstance(config, dict) and any(key in config for key in ("enable", "config"))

    @staticmethod
    def _build_env_monitor_config(config: MonitorConfig | dict[str, Any]) -> MonitorConfig:
        if isinstance(config, MonitorConfig):
            return config
        return MonitorConfig(**config)

    @staticmethod
    def _build_monitor_from_config_data(
        name: str,
        monitor_cls: type[BaseMonitor],
        config: BaseMonitorConfig | dict[str, Any] | None,
    ) -> dict[str, Any]:
        if config is None:
            config_data: dict[str, Any] = {}
        elif isinstance(config, dict):
            config_data = dict(config)
        elif is_dataclass(config):
            config_data = asdict(config)
        else:
            raise TypeError(f"Expected monitor config or dict, got {type(config).__name__}")
        config_data.setdefault("type", monitor_cls.monitor_type())
        config_data.setdefault("name", name)
        return config_data

    def get_monitor(self, name: str = "default") -> BaseMonitor | None:
        """Return a monitor by name if it exists."""
        return self._monitors.get(name)

    def get_observation(self, folder: str | Path, monitor_name: str = "default") -> dict[str, Any]:
        """Collect global observations from a monitor."""
        monitor = self.create_monitor(monitor_name)
        if monitor is None:
            return {"monitor": monitor_name, "envs": {}, "images": {}, "main_image": None}
        return monitor.get_observation(folder)

    def reset(self) -> None:
        """Remove all monitors from the manager."""
        self._monitors.clear()

    @property
    def monitors(self) -> dict[str, BaseMonitor]:
        """Return a shallow copy of monitors."""
        return dict(self._monitors)


def get_monitor_manager() -> MonitorManager:
    """Return the process-global monitor manager."""
    return MonitorManager.instance()
