"""Pipeline — ordered list of jobs that transform a policy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from cap_general.core.base import RegisteredBase
from cap_general.core.pipeline.job.base_job import BaseJob
from cap_general.core.policy import BasePolicy
from cap_general.core.utils.config import build_dataclass_config

if TYPE_CHECKING:
    from cap_general.core.robot import BaseRobot


@dataclass
class BasePipelineConfig:
    """Configuration for a pipeline."""

    jobs: list[dict[str, Any]] = field(default_factory=list)
    export_dir: str | Path = "outputs/pipeline"


class BasePipeline(RegisteredBase):
    """A set of jobs that can transform a policy.

    The pipeline does **not** own a policy.  Policies live on the agent and are
    passed in at execution time.  This keeps inference (policy.run) decoupled
    from training/production jobs (pipeline.execute).
    """

    _registry: ClassVar[dict[str, type[BasePipeline]]] = {}
    registry_key_attr: ClassVar[str] = "pipeline_type"
    pipeline_type: ClassVar[str] = "base"
    config_cls: ClassVar[type[BasePipelineConfig]] = BasePipelineConfig

    def __init__(self, config: BasePipelineConfig, logger: logging.Logger) -> None:
        self._logger = logger
        self._export_dir = Path(config.export_dir).expanduser().resolve()
        self._jobs: dict[str, BaseJob] = {}
        for job_cfg in config.jobs:
            cfg = dict(job_cfg)
            job_group, job_type = cfg.pop("type", "::").split("::", 1)
            job_config = dict(cfg.pop("config", {}))
            job_config.setdefault("export_dir", str(self._export_dir / job_group))
            job = BaseJob.create(job_group, job_type, job_config, logger=logger)
            self._jobs[job_group] = job

    # ------------------------------------------------------------------
    # Public API

    def _get_job(self, job_group: str) -> BaseJob:
        """Return the job registered under *job_group*."""
        job = self._jobs.get(job_group)
        if job is None:
            raise KeyError(f"Unknown job_group: {job_group!r}. Known: {list(self._jobs)}")
        return job

    def execute(
        self,
        jobs: list[str],
        policy: BasePolicy,
        robot: BaseRobot,
        options: dict[str, Any] | None = None,
    ) -> tuple[BasePolicy, dict[str, dict]]:
        """Run all *jobs* in order, rebuild the policy when a job returns a config dict, and return ``(policy, reports)``.

        Each job returns ``(policy_config_dict, report)``.  When ``policy_config_dict``
        is not ``None``, the policy is rebuilt via ``BasePolicy.from_config`` +
        ``policy.reset()``.  All per-job reports are collected into a
        ``{job_group: report}`` dict returned alongside the final policy.
        """
        options = dict(options or {})
        report: dict[str, dict] = {}
        for job_group in jobs:
            job = self._get_job(job_group)
            policy_dict, job_report = job.execute(policy, robot, options.get(job_group))
            if policy_dict is not None:
                policy = BasePolicy.from_config(policy_dict, logger=self._logger)
                policy.reset()
            report[job_group] = job_report
        return policy, report

    def options_doc(self) -> str:
        """Return concatenated per-job options documentation."""
        parts = []
        for job_group, job in self._jobs.items():
            doc = job.options_doc()
            if doc:
                indented = "\n".join(f"  {ln}" for ln in doc.splitlines())
                parts.append(f"[{job_group}]\n{indented}")
        return "\n".join(parts) if parts else "No job options configured."

    # ------------------------------------------------------------------
    # Construction helpers

    @classmethod
    def from_config(cls, config: Any, logger: logging.Logger, **kwargs: Any) -> BasePipeline:
        """Build a BasePipeline from a config dict (``type`` defaults to ``"base"``)."""
        if not isinstance(config, dict):
            raise TypeError(f"Expected dict config, got {type(config).__name__}")
        config_data = dict(config)
        registered_type = config_data.pop("type", cls.pipeline_type)
        subclass = cls._registry.get(registered_type, cls)
        config_obj = build_dataclass_config(subclass.config_cls, config_data)
        return subclass(config=config_obj, logger=logger)

    def __str__(self) -> str:
        job_strs = ", ".join(str(j) for j in self._jobs.values())
        return f"BasePipeline(jobs=[{job_strs}])"


BasePipeline.register()(BasePipeline)
