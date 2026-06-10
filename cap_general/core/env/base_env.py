"""Base classes for Gymnasium-style environment control loops."""

from abc import abstractmethod
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


class EnvBase(RegisteredBase, Env):
    """Abstract base class for low-level control environments."""

    _registry: ClassVar[dict[str, type["EnvBase"]]] = {}
    registry_key_method: ClassVar[str] = "env_type"

    @classmethod
    @abstractmethod
    def env_type(cls) -> str:
        """Return the registry key for this environment."""
        pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        self._step_cnt = 0
        return self._reset(seed=seed, options=options)

    @abstractmethod
    def _reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return the initial observation and info."""
        raise NotImplementedError

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one environment step.

        Returns:
            observation, reward, terminated, truncated, info.
        """
        self._step_cnt += 1
        return self._step(action)

    @abstractmethod
    def _step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take one environment step.

        Returns:
            observation, reward, terminated, truncated, info.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> ObsType:
        """Get the current observation."""
        raise NotImplementedError

    @property
    def step_cnt(self) -> int:
        """Get the current step count."""
        return self._step_cnt
