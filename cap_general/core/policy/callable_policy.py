"""Callable policy implementation."""

from typing import Callable

from cap_general.core.policy.base_policy import PolicyResult, PolicyBase


@PolicyBase.register()
class CallablePolicy(PolicyBase):
    """A policy that uses a callable function to generate code."""

    name = "Callable Policy"

    def __init__(
        self,
        generator_fn: Callable[[str], str],
        policy_name: str = "CallablePolicy",
    ):
        """Initialize with a generator function."""
        self._generator_fn = generator_fn
        self._policy_name = policy_name

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
