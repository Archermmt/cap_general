"""Job sub-package — all pipeline job types."""

from cap_general.core.pipeline.job.base_job import BaseJob, BaseJobConfig
from cap_general.core.pipeline.job.train_job import TrainJob, TrainJobConfig

__all__ = [
    "BaseJob",
    "BaseJobConfig",
    "TrainJob",
    "TrainJobConfig",
]
