"""Base monitor for collecting observations from multiple environments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core.base import RegisteredBase
from cap_general.core.utils import save_image


@dataclass
class BaseMonitorConfig:
    """Configuration for constructing a monitor."""

    name: str = "default"


class BaseMonitor(RegisteredBase):
    """Collect global observations from one or more env instances."""

    _registry: ClassVar[dict[str, type["BaseMonitor"]]] = {}
    config_cls: ClassVar[type[BaseMonitorConfig]] = BaseMonitorConfig
    registry_key_method: ClassVar[str] = "monitor_type"

    @classmethod
    def monitor_type(cls) -> str:
        """Return the registry key for this monitor."""
        return "base_monitor"

    def __init__(
        self,
        config: BaseMonitorConfig | None = None,
        logger: logging.Logger | None = None,
    ):
        self._config = config or BaseMonitorConfig()
        self._name = self._config.name
        self._logger = logger or logging.getLogger(__name__)
        self._envs: list[Any] = []

    def bind_env(self, env: Any) -> None:
        """Bind an environment instance to this monitor."""
        if env not in self._envs:
            self._envs.append(env)

    def get_observation(self, folder: str | Path) -> dict:
        """Collect monitor and bound-env observations."""
        root = Path(folder)
        root.mkdir(parents=True, exist_ok=True)
        monitor_images = self._save_monitor_obs(root)
        env_obs: dict[str, Any] = {}
        images = dict(monitor_images)
        main_image = next(iter(monitor_images.values()), None)

        for idx, env in enumerate(self._envs):
            env_name = f"env_{idx}"
            obs = env.get_observation(root / env_name)
            env_obs[env_name] = obs
            if main_image is None:
                main_image = obs.get("main_image")
            for image_name, image_path in (obs.get("images") or {}).items():
                images[f"{env_name}.{image_name}"] = image_path

        return {
            "monitor": self._name,
            "envs": env_obs,
            "images": images,
            "main_image": main_image,
        }

    def _save_monitor_obs(self, folder: str | Path) -> dict[str, Any]:
        image_dir = Path(folder)
        image_dir.mkdir(parents=True, exist_ok=True)
        images = {}
        for name, image in self._get_monitor_obs().items():
            images[name] = save_image(image_dir / f"{name}.png", image)
        return images

    def _get_monitor_obs(self) -> dict[str, Any]:
        """Return raw monitor image observations keyed by image name."""
        return {}

    @property
    def envs(self) -> list[Any]:
        """Return a shallow copy of bound envs."""
        return list(self._envs)


BaseMonitor._registry[BaseMonitor.monitor_type()] = BaseMonitor
