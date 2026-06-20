"""Base classes for Gymnasium-style environment control loops."""

import logging
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, SupportsFloat

try:
    from gymnasium import Env
except ImportError:  # pragma: no cover - fallback for minimal test environments

    class Env:
        """Minimal fallback matching the Gymnasium Env reset hook."""

        def reset(self, options: dict[str, Any] | None = None):
            self.np_random = None


from cap_general.core.base import RegisteredBase
from cap_general.core.utils import ActType, ObsType, ResetLevel, save_image, save_video
from cap_general.core.scene.context import get_current_scene


@dataclass
class BaseEnvConfig:
    """Configuration for constructing an environment."""

    seed: int | None = None
    reset_time: float = 5.0
    video_fmt: str = ""
    image_keys: list[str] = field(default_factory=list)


class BaseEnv(RegisteredBase, Env):
    """Abstract base class for low-level control environments."""

    _registry: ClassVar[dict[str, type["BaseEnv"]]] = {}
    config_cls: ClassVar[type[BaseEnvConfig]] = BaseEnvConfig
    registry_key_method: ClassVar[str] = "env_type"

    @classmethod
    def env_type(cls) -> str:
        """Return the registry key for this environment."""
        return "base_env"

    def __init__(self, config: BaseEnvConfig, logger: logging.Logger | None = None):
        self._logger = logger or logging.getLogger(__name__)
        self._seed = config.seed
        self._reset_time = config.reset_time
        self._video_fmt = config.video_fmt
        self._image_keys = list(config.image_keys or [])
        self._video_frames: dict[str, list[Any]] = {key: [] for key in self._image_keys}
        self._scene = get_current_scene()
        self._step_cnt = 0
        self._last_obs: ObsType | None = None

    @property
    def cap_scene(self) -> Any | None:
        """Return the top-level CAP scene that owns this env."""
        return self._scene

    def reset(self, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        reset_level = ResetLevel((options or {}).get("reset_level", ResetLevel.AGENT))
        if reset_level >= ResetLevel.AGENT:
            self._step_cnt = 0
            self._video_frames = {key: [] for key in self._image_keys}
        self._last_obs, info = self._reset(options=options)
        if self._reset_time > 0:
            time.sleep(self._reset_time)
        return self._last_obs, info

    @abstractmethod
    def _reset(self, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        raise NotImplementedError

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one environment step.

        Returns:
            observation, reward, terminated, truncated, info.
        """
        self._step_cnt += 1
        self._last_obs, _reward, terminated, truncated, info = self._step(action)
        reward = self.compute_reward()
        self._record_frame(self._last_obs)
        return self._last_obs, reward, terminated, truncated, info

    def get_observation(self, folder: str | Path) -> dict:
        """Return the last observation returned by step()."""
        images = {}
        if self._image_keys and isinstance(self._last_obs, dict):
            image_dir = Path(folder)
            image_dir.mkdir(parents=True, exist_ok=True)
            for image_key in self._image_keys:
                image = self._last_obs.get(image_key)
                if image is None:
                    continue
                image_path = image_dir / f"{image_key}_{self._step_cnt}.png"
                images[image_key] = save_image(image_path, image)
        main_image = images.get(self._image_keys[0]) if self._image_keys else None
        obs = {"images": images, "main_image": main_image, **self._normalize_states()}
        return obs

    def record(
        self, folder: str | Path, start_frm: int = 0, end_frm: int | None = None
    ) -> dict[str, list[Path] | Path | None]:
        """Record environment artifacts to ``folder`` and return saved paths."""
        if not self._video_fmt or not self._video_frames:
            return {"videos": [], "main_video": None}
        record_path = Path(folder)
        record_path.mkdir(parents=True, exist_ok=True)
        videos = {}
        for image_key, frames in self._video_frames.items():
            if not frames:
                continue
            end = len(frames) - 1 if end_frm is None else min(end_frm, len(frames) - 1)
            selected_frames = frames[start_frm : end + 1]
            if not selected_frames:
                continue
            video_path = record_path / f"{image_key}_{start_frm}_{end}.{self._video_fmt}"
            videos[image_key] = save_video(video_path, selected_frames)
        main_video = videos.get(self._image_keys[0]) if self._image_keys else None
        return {"videos": videos, "main_video": main_video}

    def _normalize_states(self) -> dict:
        """Return normalized non-image state values."""
        return {}

    def _record_frame(self, obs: ObsType) -> None:
        if not self._video_fmt or not self._image_keys:
            return
        if not isinstance(obs, dict):
            return
        for key in self._image_keys:
            frame = obs.get(key)
            if frame is not None:
                self._video_frames.setdefault(key, []).append(frame)

    @property
    def logger(self) -> logging.Logger:
        """Shared logger for this environment."""
        return self._logger

    @abstractmethod
    def _step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one environment step.

        Returns:
            observation, reward placeholder, terminated, truncated, info.
            The public ``step`` method computes the returned reward with
            ``compute_reward()``.
        """
        raise NotImplementedError

    def compute_reward(self) -> SupportsFloat:
        """Compute reward after a low-level step."""
        return 0.0

    @property
    def step_cnt(self) -> int:
        """Get the current step count."""
        return self._step_cnt

    @property
    def last_obs(self) -> ObsType:
        """Get the last observation."""
        return self._last_obs
