"""Reset-related enums."""

from enum import Enum, IntEnum


class ResetNamespace:
    """Namespace for reset semantics shared by agents and environments."""

    class ResetMode(str, Enum):
        """When an agent should reset its environment."""

        NEVER = "never"
        PER_EXEC = "per_exec"
        PER_TRIAL = "per_trial"

        @classmethod
        def _missing_(cls, value):
            legacy_values = {
                "execute": cls.PER_EXEC,
                "exec": cls.PER_EXEC,
                "trial": cls.PER_TRIAL,
            }
            if isinstance(value, str):
                return legacy_values.get(value)
            return None

    class EnvLevel(IntEnum):
        """Agent reset scope."""

        ROBOT = 0
        ENV = 1
        AGENT = 2


Reset = ResetNamespace
ResetMode = ResetNamespace.ResetMode
ResetFrequency = ResetNamespace.ResetMode
ResetLevel = ResetNamespace.EnvLevel
EnvResetLevel = ResetNamespace.EnvLevel
