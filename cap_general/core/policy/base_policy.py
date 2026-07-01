"""Base classes for policies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from cap_general.core.base import RegisteredBase
from cap_general.core.graph.cap_graph import CapGraph
from cap_general.core.operator.base_operator import BaseOperator
from cap_general.core.policy.policy_result import PolicyResult


@dataclass
class BasePolicyConfig:
    """Configuration for constructing a policy."""

    name: str | None = field(default=None, kw_only=True)
    describe: str = field(
        default="Generic policy interface for local model inference.",
        kw_only=True,
    )
    graph: dict[str, Any] = field(default_factory=dict, kw_only=True)


class BasePolicy(RegisteredBase):
    """DAG-based policy executor."""

    _registry: ClassVar[dict[str, type["BasePolicy"]]] = {}
    registry_key_attr: ClassVar[str] = "policy_type"
    policy_type: ClassVar[str] = "base"
    config_cls: ClassVar[type[BasePolicyConfig]] = BasePolicyConfig

    def __init__(self, config: BasePolicyConfig, logger: logging.Logger):
        self._config, self._logger = config, logger
        self._name = config.name or type(self).__name__
        self._training = False
        self._stage: str = "inference"
        self._graph: CapGraph | None = None
        self._operators: dict[str, BaseOperator] = {}

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"name={self._name!r}, "
            f"type={type(self).policy_type!r}, "
            f"training={self._training}"
            f")\n{self._graph}"
        )

    def set_stage(self, stage: str) -> None:
        """Set the active execution stage for this policy and all initialized operators."""
        self._stage = stage
        for op in self._operators.values():
            op.set_stage(stage)

    def run(self, stage: str, inputs: dict[str, Any]) -> PolicyResult:
        """Execute *stage* across all graph nodes and return a PolicyResult."""
        self.set_stage(stage)
        last_output: dict[str, Any] = {}
        for node in self._graph.nodes():
            op = self._operators[node.name]
            if not node.inputs:
                op_inputs = dict(inputs)
            else:
                op_inputs: dict[str, Any] = {}
                for _, (parent_node, output_idx) in sorted(node.inputs.items()):
                    op_inputs.update(parent_node.output(output_idx).values)
            last_output = op.run(op_inputs)
            node.output(0).update(last_output)

        return PolicyResult(code="success", policy_name=self._name, output=last_output)

    def _create_operators(self) -> None:
        for node in self._graph.nodes():
            self._operators[node.name] = BaseOperator.create(node.op_group, node.op_type, node.config, self._logger)
            self._operators[node.name].reset()

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Rebuild the graph and re-instantiate all operators."""
        self._operators = {}
        self._graph = CapGraph.from_dict(self._config.graph)
        self._create_operators()

    def get_model(self, node_name: str | None = None) -> Any:
        """Return the trainable model from the named node's operator, or the first one found."""
        if node_name is not None:
            op = self._operators.get(node_name)
            return op.get_model() if op is not None else None
        for op in self._operators.values():
            model = op.get_model()
            if model is not None:
                return model
        return None

    def visualize(self, record_dir: str | Path) -> str:
        """Save a graphviz rendering of the policy graph to *record_dir*."""
        return self._graph.visualize(record_dir, name=self._name)

    def train(self) -> "BasePolicy":
        """Switch policy and all initialized operators to training mode."""
        self._training = True
        for op in self._operators.values():
            op.train()
        self._on_train()
        return self

    def eval(self) -> "BasePolicy":
        """Switch policy and all initialized operators to evaluation mode."""
        self._training = False
        for op in self._operators.values():
            op.eval()
        self._on_eval()
        return self

    def _on_train(self) -> None:
        """Hook called after entering training mode."""

    def _on_eval(self) -> None:
        """Hook called after entering evaluation mode."""

    @property
    def describe(self) -> str:
        """Return the configured policy capability description."""
        return self._config.describe

    @property
    def logger(self) -> logging.Logger:
        """Shared logger for this policy."""
        return self._logger

    @property
    def name(self) -> str:
        """Return the name of this policy."""
        return self._name

    @property
    def training(self) -> bool:
        """Whether the policy is in training mode."""
        return self._training


BasePolicy._registry["base"] = BasePolicy
