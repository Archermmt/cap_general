"""BaseOperator: execution unit bound to a single CapNode."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, ClassVar

from cap_general.core.utils.config import build_dataclass_config


def to_stage_fn(func: Callable[..., Any]) -> Callable[..., Any]:
    """Mark an operator method as callable through :meth:`BaseOperator.run`."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    wrapper._is_stage_func = True
    return wrapper


@dataclass
class BaseOperatorConfig:
    """Minimal base config for operators (subclasses may define richer configs)."""


class BaseOperator:
    """Base class for all operator components."""

    _registry: ClassVar[dict[str, dict[str, type["BaseOperator"]]]] = {}
    op_group: ClassVar[str] = "base"
    op_type: ClassVar[str] = "base"
    config_cls: ClassVar[type | None] = None

    @classmethod
    def register(cls):
        """Decorator that registers a subclass by ``op_group`` and ``op_type``."""

        def decorator(subclass: type[BaseOperator]) -> type[BaseOperator]:
            og = getattr(subclass, "op_group", None)
            ot = getattr(subclass, "op_type", None)
            if not og or not ot:
                raise TypeError(f"{subclass.__name__} must define both op_group and op_type")
            cls._registry.setdefault(og, {})[ot] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, op_group: str, op_type: str, config: dict[str, Any], logger: logging.Logger) -> BaseOperator:
        """Instantiate a registered operator by group and type."""
        op_cls = cls._registry.get(op_group, {}).get(op_type)
        if op_cls is None:
            raise KeyError(f"Unknown operator: {op_group}::{op_type}")
        return op_cls(config=config, logger=logger)

    @classmethod
    def get_registered_class(cls, op_group: str, op_type: str) -> type[BaseOperator] | None:
        """Return a registered class, or ``None`` if not found."""
        return cls._registry.get(op_group, {}).get(op_type)

    @classmethod
    def registered_types(cls, op_group: str | None = None) -> dict[str, list[str]] | list[str]:
        """Return registered types."""
        if op_group is not None:
            return list(cls._registry.get(op_group, {}).keys())
        return {g: list(types.keys()) for g, types in cls._registry.items()}

    def __init__(self, config: dict[str, Any], logger: logging.Logger) -> None:
        cls = type(self)
        self._config = build_dataclass_config(cls.config_cls, config) if cls.config_cls is not None else config
        self._logger = logger
        self._stage: str = "inference"
        self._training: bool = False
        self._stage_funcs = {
            name: getattr(self, name)
            for name in dir(type(self))
            if getattr(getattr(type(self), name, None), "_is_stage_func", False)
        }

    def reset(self) -> None:
        """Initialize or reset operator resources."""

    def set_stage(self, stage: str) -> None:
        """Set the active execution stage."""
        self._stage = stage

    def train(self) -> None:
        """Switch operator to training mode."""
        self._training = True
        self._on_train()

    def eval(self) -> None:
        """Switch operator to evaluation mode."""
        self._training = False
        self._on_eval()

    def _on_train(self) -> None:
        """Hook called after entering training mode."""

    def _on_eval(self) -> None:
        """Hook called after entering evaluation mode."""

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute the current stage with *inputs* and return a dict output."""
        method = self._stage_funcs.get(self._stage)
        if method is None:
            raise AttributeError(f"{type(self).__name__} has no stage {self._stage!r}")
        result = method(inputs)
        return result if isinstance(result, dict) else {"output": result}

    def get_model(self) -> Any:
        """Return the underlying trainable model, or None if not applicable."""
        return None

    def cleanup(self) -> None:
        """Release resources."""
