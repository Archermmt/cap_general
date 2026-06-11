"""Static policy implementation."""

from dataclasses import dataclass

from cap_general.core.policy.base_policy import PolicyResult, PolicyBase


@dataclass
class StaticPolicyConfig:
    """Configuration for StaticPolicy."""

    code: str


@PolicyBase.register()
class StaticPolicy(PolicyBase):
    """A policy that always returns the same fixed code."""

    name = "Static Policy"
    config_cls = StaticPolicyConfig

    def __init__(self, code: str | None = None, config: StaticPolicyConfig | None = None):
        """Initialize with fixed code."""
        config = config or StaticPolicyConfig(code=code or "")
        self._code = config.code

    @classmethod
    def policy_type(cls) -> str:
        return "static"

    def inference(self, prompt: str) -> PolicyResult:
        """Return the fixed code regardless of prompt."""
        return PolicyResult(code=self._code, policy_name=self.policy_name)

    @property
    def policy_name(self) -> str:
        return "StaticPolicy"
