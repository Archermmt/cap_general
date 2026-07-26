"""CapNode: a single node in a CapGraph, bound to one operator type."""

from __future__ import annotations

from typing import Any

from cap_general.core.policy.graph.cap_data import CapData


class CapNode:
    """A computation node in a CapGraph."""

    def __init__(self, name: str, node_type: str, config: dict[str, Any] | None = None) -> None:
        self._name = name
        parts = node_type.split("::", 1)
        self._op_group: str = parts[0]
        self._op_type: str = parts[1] if len(parts) == 2 else ""
        self._config: dict[str, Any] = config or {}
        self._parents: list[CapNode] = []
        self._children: list[CapNode] = []
        self._inputs: dict[int, tuple[CapNode, int]] = {}
        self._outputs: dict[int, CapData] = {0: CapData(name=f"{self._name}:0")}

    def __str__(self) -> str:
        parents = ",".join(p.name for p in self._parents)
        children = ",".join(c.name for c in self._children)
        info = f"{self._name}({self._op_group}/{self._op_type})<P:{parents}| C:{children}>"
        if self._config:
            cfg = " ".join(f"{k}={v}" for k, v in self._config.items())
            info += f"\n  CONFIG: {cfg}"
        if self._inputs:
            refs = "| ".join(f"{idx}->{p.name}:{ref}" for idx, (p, ref) in sorted(self._inputs.items()))
            info += f"\n  IN: {refs}"
        if self._outputs:
            refs = "| ".join(f"{idx}->{data.name}" for idx, data in sorted(self._outputs.items()))
            info += f"\n  OUT: {refs}"
        return info

    # ------------------------------------------------------------------
    # Public read-only properties

    @property
    def name(self) -> str:
        return self._name

    @property
    def op_group(self) -> str:
        return self._op_group

    @property
    def op_type(self) -> str:
        return self._op_type

    @property
    def node_type(self) -> str:
        return f"{self._op_group}::{self._op_type}" if self._op_type else self._op_group

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @staticmethod
    def is_group(obj: Any, op_group: str) -> bool:
        """Return True if *obj* (CapNode, BaseJob, or type dict) belongs to *op_group*."""
        if isinstance(obj, dict):
            return obj.get("type", "").split("::", 1)[0] == op_group
        for attr in ("op_group", "job_group"):
            val = getattr(obj, attr, None)
            if val is not None:
                return val == op_group
        return False

    @staticmethod
    def is_type(obj: Any, op_group: str, op_type: str) -> bool:
        """Return True if *obj* (CapNode, BaseJob, or type dict) matches *op_group*::*op_type*."""
        if isinstance(obj, dict):
            parts = obj.get("type", "").split("::", 1)
            return len(parts) == 2 and parts[0] == op_group and parts[1] == op_type
        for group_attr, type_attr in (("op_group", "op_type"), ("job_group", "job_type")):
            val = getattr(obj, group_attr, None)
            if val is not None:
                return val == op_group and getattr(obj, type_attr, None) == op_type
        return False

    @property
    def parents(self) -> list[CapNode]:
        return self._parents

    @property
    def children(self) -> list[CapNode]:
        return self._children

    @property
    def inputs(self) -> dict[int, tuple[CapNode, int]]:
        return self._inputs

    @property
    def outputs(self) -> dict[int, CapData]:
        return self._outputs

    # ------------------------------------------------------------------
    # Graph wiring

    def add_parent(self, node: "CapNode") -> None:
        if node not in self._parents:
            self._parents.append(node)
        if self not in node._children:
            node._children.append(self)

    def add_child(self, node: "CapNode") -> None:
        node.add_parent(self)

    def remove_parent(self, node: "CapNode") -> None:
        if node in self._parents:
            self._parents.remove(node)
        if self in node._children:
            node._children.remove(self)
        for idx, (parent, _ref) in list(self._inputs.items()):
            if parent is node:
                self._inputs.pop(idx)

    def remove_child(self, node: "CapNode") -> None:
        node.remove_parent(self)

    def parent(self, idx: int) -> CapNode:
        return self._parents[idx]

    def add_input(self, idx: int, node: "CapNode", output_idx: int = 0) -> None:
        self._inputs[idx] = (node, output_idx)
        self.add_parent(node)
        node.ensure_output(output_idx)

    def input(self, idx: int) -> CapData:
        parent, output_idx = self._inputs[idx]
        return parent.output(output_idx)

    def get_inputs(self) -> dict[int, CapData]:
        return {idx: self.input(idx) for idx in sorted(self._inputs)}

    def ensure_output(self, idx: int = 0) -> CapData:
        if idx not in self._outputs:
            self._outputs[idx] = CapData(name=f"{self._name}:{idx}")
        return self._outputs[idx]

    def output(self, idx: int = 0) -> CapData:
        return self.ensure_output(idx)

    def get_outputs(self) -> dict[int, CapData]:
        return {idx: self.output(idx) for idx in sorted(self._outputs)}

    def to_dict(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "name": self._name,
            "type": self.node_type,
            "config": self._config,
        }
        if self._inputs:
            info["inputs"] = {idx: f"{p.name}:{ref}" for idx, (p, ref) in sorted(self._inputs.items())}
        return info
