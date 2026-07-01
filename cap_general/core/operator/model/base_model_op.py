"""Model operator base class."""

from __future__ import annotations

from cap_general.core.operator.base_operator import BaseOperator


class ModelOp(BaseOperator):
    """Base class for model-type operators (op_group = "model")."""

    op_group: str = "model"
    op_type: str = "base"
