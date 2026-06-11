"""Base classes for Gymnasium-style environment control loops."""

import io
import time
from abc import abstractmethod
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, SupportsFloat, TypeVar

try:
    from gymnasium import Env
except ImportError:  # pragma: no cover - fallback for minimal test environments

    class Env:
        """Minimal fallback matching the Gymnasium Env reset hook."""

        def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
            self.np_random = None


from cap_general.core.base import RegisteredBase

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class BaseEnvConfig:
    """Configuration for constructing an environment."""

    env_type: str = "base_env"
    reset_time: float = 2.0
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

    def __init__(self, config: BaseEnvConfig | None = None):
        config = config or BaseEnvConfig()
        self._reset_time = config.reset_time
        self._video_fmt = config.video_fmt
        self._image_keys = list(config.image_keys or [])
        self._video_frames: dict[str, list[Any]] = {key: [] for key in self._image_keys}
        self._step_cnt = 0
        self._last_obs: ObsType | None = None

    @classmethod
    def from_config(cls, config: "BaseEnvConfig | dict[str, Any]") -> "BaseEnv":
        """Instantiate a registered environment from config."""
        config_data = cls._normalize_component_config(config, "env_type")
        env_type = config_data.pop("env_type")
        env_cls = cls.get_registered_type(env_type)
        if env_cls is None:
            raise KeyError(f"Unknown env type: {env_type}")

        config_cls = getattr(env_cls, "config_cls", None)
        if config_cls is not None and is_dataclass(config_cls):
            config_obj = cls._build_dataclass_config(config_cls, config_data)
            return env_cls(config=config_obj)
        return env_cls(**config_data)

    @staticmethod
    def _normalize_component_config(
        config: "BaseEnvConfig | dict[str, Any]",
        type_key: str,
    ) -> dict[str, Any]:
        if is_dataclass(config):
            return dict(config.__dict__)
        if not isinstance(config, dict):
            raise TypeError(f"Expected config dict, got {type(config).__name__}")

        data = dict(config.get("config", {}))
        for key, value in config.items():
            if key != "config":
                data[key] = value
        if "type" in data and type_key not in data:
            data[type_key] = data.pop("type")
        if type_key not in data:
            raise KeyError(f"Missing config field: {type_key}")
        return data

    @staticmethod
    def _build_dataclass_config(config_cls, config_data: dict[str, Any]):
        field_names = {field.name for field in fields(config_cls)}
        values = {key: value for key, value in config_data.items() if key in field_names}
        return config_cls(**values)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        pose_only: bool = False,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        self._step_cnt = 0
        self._video_frames = {key: [] for key in self._image_keys}
        obs, info = self._reset(seed=seed, options=options, pose_only=pose_only)
        self._last_obs = obs
        if self._reset_time > 0:
            time.sleep(self._reset_time)
        return obs, info

    @abstractmethod
    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        pose_only: bool = False,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        raise NotImplementedError

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one environment step.

        Returns:
            observation, reward, terminated, truncated, info.
        """
        self._step_cnt += 1
        obs, reward, terminated, truncated, info = self._step(action)
        self._last_obs = obs
        self._record_frame(obs)
        return obs, reward, terminated, truncated, info

    def _record_frame(self, obs: ObsType) -> None:
        if not self._video_fmt or not self._image_keys:
            return
        if not isinstance(obs, dict):
            return
        for key in self._image_keys:
            frame = obs.get(key)
            if frame is not None:
                self._video_frames.setdefault(key, []).append(frame)

    def get_observation(self, images_only: bool = False) -> dict | ObsType:
        """Return the current observation."""
        if images_only:
            if not self._image_keys or not isinstance(self._last_obs, dict):
                return {"images": {}, "main_image": None}
            images = {key: self._last_obs[key] for key in self._image_keys if key in self._last_obs}
            return {"images": images, "main_image": images.get(self._image_keys[0])}
        return self._last_obs

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
            video_path = record_path / f"{image_key}_{start_frm:06d}_{end:06d}.{self._video_fmt}"
            videos[image_key] = str(self._save_video(video_path, selected_frames))
        return {"videos": videos, "main_video": videos[self._image_keys[0]]}

    @classmethod
    def _save_video(cls, path: Path, frames: list[Any]) -> Path:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("Saving video requires imageio") from exc
        iio.imwrite(path, [cls._frame_to_array(frame) for frame in frames])
        return path

    @staticmethod
    def _frame_to_array(frame: Any):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Saving video frames requires pillow and numpy") from exc

        if isinstance(frame, bytes):
            frame = Image.open(io.BytesIO(frame))
        if hasattr(frame, "convert"):
            frame = frame.convert("RGB")

        array = np.asarray(frame)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array

    @abstractmethod
    def _step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one environment step.

        Returns:
            observation, reward, terminated, truncated, info.
        """
        raise NotImplementedError

    @property
    def step_cnt(self) -> int:
        """Get the current step count."""
        return self._step_cnt
