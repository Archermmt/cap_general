"""Operator module.

Importing this package registers all built-in operators so they are
available via ``BaseOperator.create()`` and ``BaseOperator.get_registered_class()``.
"""

from cap_general.core.operator.base_operator import BaseOperator, BaseOperatorConfig, to_stage_fn
from cap_general.core.operator.model import ModelOp

__all__ = [
    "BaseOperator",
    "BaseOperatorConfig",
    "ModelOp",
    "to_stage_fn",
]
