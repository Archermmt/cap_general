"""Model operator group.

Importing this sub-package triggers registration of all model-type operators.
"""

from cap_general.core.operator.model.base_model_op import ModelOp

import cap_general.core.operator.model.graspnet_op  # noqa: F401
import cap_general.core.operator.model.huggingface_op  # noqa: F401
import cap_general.core.operator.model.pyroki_op  # noqa: F401
import cap_general.core.operator.model.rsl_rl_op  # noqa: F401
import cap_general.core.operator.model.sam3_op  # noqa: F401
import cap_general.core.operator.model.starvla_op  # noqa: F401

__all__ = ["ModelOp"]
