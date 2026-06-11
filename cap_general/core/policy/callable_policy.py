"""Callable policy implementation."""

from dataclasses import dataclass
from typing import Callable

from cap_general.core.policy.base_policy import PolicyResult, PolicyBase


@dataclass
class CallablePolicyConfig:
    """Configuration for CallablePolicy."""

    generator_fn: Callable[[str], str] | None = None
    policy_name: str = "CallablePolicy"


@PolicyBase.register()
class CallablePolicy(PolicyBase):
    """A policy that uses a callable function to generate code."""

    name = "Callable Policy"
    config_cls = CallablePolicyConfig

    def __init__(
        self,
        generator_fn: Callable[[str], str] | None = None,
        policy_name: str = "CallablePolicy",
        config: CallablePolicyConfig | None = None,
    ):
        """Initialize with a generator function."""
        config = config or CallablePolicyConfig(
            generator_fn=generator_fn,
            policy_name=policy_name,
        )
        if config.generator_fn is None:
            raise ValueError("CallablePolicy requires generator_fn")
        self._generator_fn = config.generator_fn
        self._policy_name = config.policy_name

    @classmethod
    def policy_type(cls) -> str:
        return "callable"

    def inference(self, prompt: str) -> PolicyResult:
        """Run inference using the provided function."""
        code = self._generator_fn(prompt)
        return PolicyResult(code=code, policy_name=self.policy_name)

    @property
    def policy_name(self) -> str:
        return self._policy_name
