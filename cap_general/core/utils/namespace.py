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
    """How much task-execution detail to record in the agent history.

    NEVER  - No history entries are written; per-step records are not saved.
    TASK   - One response history entry per execute/retry call (role=agent);
             no LLM-request entries; no per-step record artifacts.
    ALL    - Full trace: LLM request + agent response entries for every
             execute/retry/train call, and per-step record artifacts saved.
    """

    NEVER = "never"
    TASK = "task"
    ALL = "all"
