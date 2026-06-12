"""Reset-related enums."""

from enum import Enum, IntEnum


class ResetNamespace:
    """Namespace for reset semantics shared by agents and environments."""

    class Frequency(str, Enum):
        """When an agent should reset its environment."""

        NEVER = "never"
        EXECUTE = "execute"
        TRIAL = "trial"

    class EnvLevel(IntEnum):
        """Agent reset scope."""

        ROBOT = 0
        ENV = 1
        AGENT = 2


Reset = ResetNamespace
ResetFrequency = ResetNamespace.Frequency
ResetLevel = ResetNamespace.EnvLevel
EnvResetLevel = ResetNamespace.EnvLevel
