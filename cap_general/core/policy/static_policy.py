"""Static policy implementation."""

from cap_general.core.policy.base_policy import PolicyResult, PolicyBase


@PolicyBase.register()
class StaticPolicy(PolicyBase):
    """A policy that always returns the same fixed code."""

    name = "Static Policy"

    def __init__(self, code: str):
        """Initialize with fixed code."""
        self._code = code

    @classmethod
    def policy_type(cls) -> str:
        return "static"

    def inference(self, prompt: str) -> PolicyResult:
        """Return the fixed code regardless of prompt."""
        return PolicyResult(code=self._code, policy_name=self.policy_name)

    @property
    def policy_name(self) -> str:
        return "StaticPolicy"
