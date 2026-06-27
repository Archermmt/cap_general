"""Base classes for policies."""

from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, ClassVar, Optional

from cap_general.core.base import RegisteredBase


@dataclass
class PolicyResult:
    """Result from policy inference."""

    code: str
    policy_name: str
    metadata: Optional[dict] = None


def apply_stop_sequences(text: str, stop: list[str] | None = None) -> str:
    """Truncate text at the earliest stop sequence."""
    if not stop:
        return text

    earliest = None
    for sequence in stop:
        if not sequence:
            continue
        index = text.find(sequence)
        if index >= 0 and (earliest is None or index < earliest):
            earliest = index

    return text if earliest is None else text[:earliest]


def normalize_prompt(prompt: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Normalize supported prompt inputs for local model backends."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return prompt
    raise TypeError(f"Unsupported prompt type: {type(prompt).__name__}")


@dataclass
class BasePolicyConfig:
    """Configuration for constructing a policy."""

    describe: str = field(
        default="Generic policy interface for local model inference.",
        kw_only=True,
    )


class BasePolicy(RegisteredBase):
    """Abstract base class for policies."""

    _registry: ClassVar[dict[str, type["BasePolicy"]]] = {}
    config_cls: ClassVar[type[BasePolicyConfig]] = BasePolicyConfig
    registry_key_method: ClassVar[str] = "policy_type"

    def __init__(self, config: BasePolicyConfig, logger: logging.Logger):
        self._config = config
        self._logger = logger

    @classmethod
    def policy_type(cls) -> str:
        """Return the registry key for this policy."""
        return "base"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Load or initialize policy resources and reset transient state."""

    @abstractmethod
    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """Run local model inference."""
        pass

    @property
    def describe(self) -> str:
        """Return the configured policy capability description."""
        return self._config.describe

    @property
    def logger(self) -> logging.Logger:
        """Shared logger for this policy."""
        return self._logger

    @property
    def name(self) -> str:
        """Return the name of this policy."""
        return self._name
