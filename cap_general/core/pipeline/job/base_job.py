"""Base class for pipeline jobs — dual-key (job_group::job_type) registration."""

from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from cap_general.core.utils.config import build_dataclass_config

if TYPE_CHECKING:
    from cap_general.core.policy import BasePolicy
    from cap_general.core.robot import BaseRobot


@dataclass
class BaseJobConfig:
    """Configuration for a pipeline job."""

    export_dir: str = field(default="outputs/pipeline")


class BaseJob:
    """Base class for all pipeline jobs.

    Registration uses two class-level keys:

    * ``job_group`` — what kind of job this is (``"train"``, ``"policy"``,
      ``"eval"``, ``"compile"`` …).
    * ``job_type`` — the concrete implementation within that category
      (``"base"``, ``"rl"``, ``"rsl_rl"`` …).

    Lookup key = ``"job_group::job_type"``.  Register subclasses with the
    ``@BaseJob.register()`` decorator.
    """

    _registry: ClassVar[dict[str, type[BaseJob]]] = {}

    job_group: ClassVar[str] = "base"
    job_type: ClassVar[str] = "base"
    config_cls: ClassVar[type[BaseJobConfig]] = BaseJobConfig

    def __init__(self, config: BaseJobConfig, logger: logging.Logger | None = None) -> None:
        self._config = config
        self._logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Registration

    @classmethod
    def register(cls) -> Callable[[type[BaseJob]], type[BaseJob]]:
        """Class decorator that registers a job subclass by ``job_group::job_type``."""

        def decorator(subclass: type[BaseJob]) -> type[BaseJob]:
            key = f"{subclass.job_group}::{subclass.job_type}"
            cls._registry[key] = subclass
            return subclass

        return decorator

    @classmethod
    def get_class(cls, job_group: str, job_type: str) -> type[BaseJob] | None:
        """Return the registered class for *job_group*::*job_type*, or ``None``."""
        return cls._registry.get(f"{job_group}::{job_type}")

    @classmethod
    def registered_keys(cls) -> list[str]:
        """Return all registered ``job_group::job_type`` keys."""
        return list(cls._registry)

    @classmethod
    def create(cls, job_group: str, job_type: str, config: dict[str, Any], **kwargs: Any) -> BaseJob:
        """Instantiate a registered job by group and type."""
        key = f"{job_group}::{job_type}"
        subclass = cls._registry.get(key)
        if subclass is None:
            raise KeyError(f"Unknown job {key!r}. Registered keys: {list(cls._registry)}")
        config_obj = build_dataclass_config(subclass.config_cls, config)
        return subclass(config=config_obj, **kwargs)

    # ------------------------------------------------------------------
    # Instance API

    @property
    def name(self) -> str:
        return f"{self.job_group}::{self.job_type}"

    @staticmethod
    def is_group(obj: Any, job_group: str) -> bool:
        """Return True if *obj* (BaseJob, CapNode, or type dict) belongs to *job_group*."""
        if isinstance(obj, dict):
            return obj.get("type", "").split("::", 1)[0] == job_group
        for attr in ("job_group", "op_group"):
            val = getattr(obj, attr, None)
            if val is not None:
                return val == job_group
        return False

    @staticmethod
    def is_type(obj: Any, job_group: str, job_type: str) -> bool:
        """Return True if *obj* (BaseJob, CapNode, or type dict) matches *job_group*::*job_type*."""
        if isinstance(obj, dict):
            parts = obj.get("type", "").split("::", 1)
            return len(parts) == 2 and parts[0] == job_group and parts[1] == job_type
        for group_attr, type_attr in (("job_group", "job_type"), ("op_group", "op_type")):
            val = getattr(obj, group_attr, None)
            if val is not None:
                return val == job_group and getattr(obj, type_attr, None) == job_type
        return False

    @abstractmethod
    def execute(
        self,
        policy: BasePolicy,
        robot: BaseRobot,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """Execute this job and return ``(policy_graph_dict, report)``.

        Return ``(None, report)`` to leave the policy unchanged, or
        ``(graph_dict, report)`` to have the pipeline rebuild the policy from
        the new graph config.
        """

    def options_doc(self) -> str:
        """Describe options this job accepts. Override in subclasses."""
        return ""

    def __str__(self) -> str:
        return f"{self.name}({self._config})"
