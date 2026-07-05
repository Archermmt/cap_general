"""Train job base — subclasses implement framework-specific training logic."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from cap_general.core.pipeline.job.base_job import BaseJob, BaseJobConfig

if TYPE_CHECKING:
    from cap_general.core.policy import BasePolicy


@dataclass
class TrainJobConfig(BaseJobConfig):
    """Configuration for a TrainJob."""

    epoch: int = 1


@BaseJob.register()
class TrainJob(BaseJob):
    """Base train job — subclasses implement ``_execute`` with the training loop.

    This class handles train/eval mode switching around the training call so
    subclasses can focus on the framework-specific logic.
    """

    job_group = "train"
    job_type = "base"
    config_cls = TrainJobConfig

    def execute(
        self,
        policy: BasePolicy,
        robot: Any,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Switch to train mode, run ``_execute``, and restore eval mode."""
        options = options or {}
        policy.train()
        robot.train()
        try:
            return self._execute(policy, robot, options)
        finally:
            policy.eval()
            robot.eval()

    @abstractmethod
    def _execute(
        self,
        policy: BasePolicy,
        robot: Any,
        options: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Implement framework-specific training and return ``(policy_config_dict, report)``."""

    def options_doc(self) -> str:
        return "epoch: training steps/epochs (default: config value)"
