"""Operator module.

Importing this package registers all built-in operators so they are
available via ``BaseOperator.create()`` and ``BaseOperator.get_registered_class()``.
"""

from cap_general.core.operator.base_operator import BaseOperator, BaseOperatorConfig, to_stage_fn
from cap_general.core.operator.model import ModelOp

import cap_general.core.operator.model.rsl_rl_op  # noqa: F401
import cap_general.core.operator.model.sam3_op  # noqa: F401
import cap_general.core.operator.model.graspnet_op  # noqa: F401
import cap_general.core.operator.model.starvla_op  # noqa: F401
import cap_general.core.operator.model.huggingface_op  # noqa: F401
import cap_general.core.operator.model.pyroki_op  # noqa: F401

__all__ = [
    "BaseOperator",
    "BaseOperatorConfig",
    "ModelOp",
    "to_stage_fn",
]
