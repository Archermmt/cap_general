"""Base classes for policies."""

from abc import abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Any, ClassVar

from cap_general.core.base import RegisteredBase


@dataclass
class PolicyResult:
    """Result from policy inference."""

    code: str
    policy_name: str
    metadata: dict | None = None


@dataclass
class BasePolicyConfig:
    """Configuration for constructing a policy."""

    name: str | None = field(default=None, kw_only=True)
    describe: str = field(
        default="Generic policy interface for local model inference.",
        kw_only=True,
    )


class BasePolicy(RegisteredBase):
    """Abstract base class for policies."""

    _registry: ClassVar[dict[str, type["BasePolicy"]]] = {}
    config_cls: ClassVar[type[BasePolicyConfig]] = BasePolicyConfig
    registry_key_method: ClassVar[str] = "policy_type"

    def __getattribute__(self, item: str) -> Any:
        if item == "name":
            return object.__getattribute__(self, "_name")
        return super().__getattribute__(item)

    def __init__(self, config: BasePolicyConfig, logger: logging.Logger):
        self._config, self._logger = config, logger
        self._name = config.name or type(self).__name__
        self._training = False

    @classmethod
    def policy_type(cls) -> str:
        """Return the registry key for this policy."""
        return "base"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Load or initialize policy resources and reset transient state."""

    def train(self) -> "BasePolicy":
        """Switch policy behavior to training mode."""
        self._training = True
        self._on_train()
        return self

    def eval(self) -> "BasePolicy":
        """Switch policy behavior to evaluation mode."""
        self._training = False
        self._on_eval()
        return self

    def _on_train(self) -> None:
        """Hook called after entering training mode."""

    def _on_eval(self) -> None:
        """Hook called after entering evaluation mode."""

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

    @property
    def training(self) -> bool:
        """Whether the policy is in training mode."""
        return self._training
