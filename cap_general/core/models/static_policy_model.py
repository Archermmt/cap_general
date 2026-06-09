"""Static policy model implementation."""

from cap_general.core.models.base_model import PolicyGenerationResult, PolicyModel


@PolicyModel.register()
class StaticPolicyModel(PolicyModel):
    """A policy model that always returns the same fixed code."""

    name = "Static Policy Model"

    def __init__(self, code: str):
        """Initialize with fixed code."""
        self._code = code

    @classmethod
    def model_type(cls) -> str:
        return "static"

    def generate(self, prompt: str) -> PolicyGenerationResult:
        """Return the fixed code regardless of prompt."""
        return PolicyGenerationResult(code=self._code, model_name=self.model_name)

    @property
    def model_name(self) -> str:
        return "StaticPolicyModel"
