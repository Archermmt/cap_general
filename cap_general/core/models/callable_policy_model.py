"""Callable policy model implementation."""

from typing import Callable

from cap_general.core.models.base_model import PolicyGenerationResult, PolicyModel


@PolicyModel.register()
class CallablePolicyModel(PolicyModel):
    """A policy model that uses a callable function to generate code."""

    name = "Callable Policy Model"

    def __init__(
        self,
        generator_fn: Callable[[str], str],
        model_name: str = "CallablePolicyModel",
    ):
        """Initialize with a generator function."""
        self._generator_fn = generator_fn
        self._model_name = model_name

    @classmethod
    def model_type(cls) -> str:
        return "callable"

    def generate(self, prompt: str) -> PolicyGenerationResult:
        """Generate code using the provided function."""
        code = self._generator_fn(prompt)
        return PolicyGenerationResult(code=code, model_name=self.model_name)

    @property
    def model_name(self) -> str:
        return self._model_name
