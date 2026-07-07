"""Evaluation job base for pipeline policy scoring."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cap_general.core.pipeline.job.base_job import BaseJob, BaseJobConfig

if TYPE_CHECKING:
    from cap_general.core.policy import BasePolicy
    from cap_general.core.robot import BaseRobot


@dataclass
class EvalJobConfig(BaseJobConfig):
    """Configuration for an EvalJob."""

    epoch: int = 1


@BaseJob.register()
class EvalJob(BaseJob):
    """Base job for evaluating a policy without updating it."""

    job_group = "eval"
    job_type = "base"
    config_cls = EvalJobConfig

    def execute(
        self,
        policy: BasePolicy,
        robot: BaseRobot,
        options: dict[str, Any] | None = None,
    ) -> tuple[None, dict[str, Any]]:
        """Run evaluation and leave both policy and robot in eval mode."""
        policy.eval()
        robot.eval()
        try:
            return self._execute(policy, robot, options or {})
        finally:
            policy.eval()
            robot.eval()

    @abstractmethod
    def _execute(
        self,
        policy: BasePolicy,
        robot: BaseRobot,
        options: dict[str, Any],
    ) -> tuple[None, dict[str, Any]]:
        """Evaluate *policy* and return ``(None, report)``."""

    def options_doc(self) -> str:
        return "epoch: number of evaluation episodes (default: config value)"
