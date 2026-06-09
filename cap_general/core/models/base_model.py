"""Base classes for policy models."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Optional

from cap_general.core.base import RegisteredBase


@dataclass
class PolicyGenerationResult:
    """Result from policy model code generation."""

    code: str
    model_name: str
    metadata: Optional[dict] = None


class PolicyModel(RegisteredBase):
    """Abstract base class for policy models that generate code from prompts."""

    _registry: ClassVar[dict[str, type["PolicyModel"]]] = {}
    registry_key_method: ClassVar[str] = "model_type"

    @classmethod
    @abstractmethod
    def model_type(cls) -> str:
        """Return the registry key for this model."""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> PolicyGenerationResult:
        """Generate code from a natural language prompt."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the model."""
        pass
