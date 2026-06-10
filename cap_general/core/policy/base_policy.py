"""Base classes for policies."""

from abc import abstractmethod
from dataclasses import dataclass
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


class PolicyBase(RegisteredBase):
    """Abstract base class for policies."""

    _registry: ClassVar[dict[str, type["PolicyBase"]]] = {}
    registry_key_method: ClassVar[str] = "policy_type"

    @classmethod
    @abstractmethod
    def policy_type(cls) -> str:
        """Return the registry key for this policy."""
        pass

    @abstractmethod
    def inference(self, *args: Any, **kwargs: Any) -> Any:
        """Run local model inference."""
        pass

    @property
    @abstractmethod
    def policy_name(self) -> str:
        """Return the name of the policy."""
        pass
