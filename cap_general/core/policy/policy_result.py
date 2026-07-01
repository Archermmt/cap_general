"""Policy execution result."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PolicyResult:
    """Result from policy inference."""

    code: str
    policy_name: str
    output: Any = None
    metadata: dict | None = None
