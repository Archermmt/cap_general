"""Reset-related enums."""

from enum import Enum, IntEnum


class ResetMode(str, Enum):
    """When an agent should reset its robot."""

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


class ResetLevel(IntEnum):
    """Agent reset scope."""

    ROBOT = 0
    AGENT = 1
    SCENE = 2


class TraceLevel(str, Enum):
    """How much task-execution detail to record in the scene history.

    NEVER  - No history entries are written; per-step records are not saved.
    TASK   - Request and response entries for execute/retry/monitor/get_obs;
             no per-step record artifacts.
    ALL    - TASK entries plus train entries and per-step record artifacts.
    """

    NEVER = "never"
    TASK = "task"
    ALL = "all"
