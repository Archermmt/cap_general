"""CapNode: a single node in a CapGraph, bound to one operator type."""

from __future__ import annotations

from typing import Any

from cap_general.core.graph.cap_data import CapData


class CapNode:
    """A computation node in a CapGraph."""

    def __init__(self, name: str, node_type: str, config: dict[str, Any] | None = None) -> None:
        self.name = name
        self.node_type = node_type
        self.config = config or {}
        self.parents: list[CapNode] = []
        self.children: list[CapNode] = []
        self.inputs: dict[int, tuple[CapNode, int]] = {}
        self.outputs: dict[int, CapData] = {0: CapData(name=f"{self.name}:0")}

    def __str__(self) -> str:
        parents = ",".join(parent.name for parent in self.parents)
        children = ",".join(child.name for child in self.children)
        info = f"{self.name}({self.op_group}/{self.op_type})<P:{parents}| C:{children}>"
        if self.config:
            cfg = " ".join(f"{k}={v}" for k, v in self.config.items())
            info += f"\n  CONFIG: {cfg}"
        if self.inputs:
            refs = "| ".join(f"{idx}->{parent.name}:{ref}" for idx, (parent, ref) in sorted(self.inputs.items()))
            info += f"\n  IN: {refs}"
        if self.outputs:
            refs = "| ".join(f"{idx}->{data.name}" for idx, data in sorted(self.outputs.items()))
            info += f"\n  OUT: {refs}"
        return info

    def add_parent(self, node: "CapNode") -> None:
        if node not in self.parents:
            self.parents.append(node)
        if self not in node.children:
            node.children.append(self)

    def add_child(self, node: "CapNode") -> None:
        node.add_parent(self)

    def remove_parent(self, node: "CapNode") -> None:
        if node in self.parents:
            self.parents.remove(node)
        if self in node.children:
            node.children.remove(self)
        for idx, (parent, _ref) in list(self.inputs.items()):
            if parent is node:
                self.inputs.pop(idx)

    def remove_child(self, node: "CapNode") -> None:
        node.remove_parent(self)

    def parent(self, idx: int) -> CapNode:
        return self.parents[idx]

    def add_input(self, idx: int, node: "CapNode", output_idx: int = 0) -> None:
        self.inputs[idx] = (node, output_idx)
        self.add_parent(node)
        node.ensure_output(output_idx)

    def input(self, idx: int) -> CapData:
        parent, output_idx = self.inputs[idx]
        return parent.output(output_idx)

    def get_inputs(self) -> dict[int, CapData]:
        return {idx: self.input(idx) for idx in sorted(self.inputs)}

    def ensure_output(self, idx: int = 0) -> CapData:
        if idx not in self.outputs:
            self.outputs[idx] = CapData(name=f"{self.name}:{idx}")
        return self.outputs[idx]

    def output(self, idx: int = 0) -> CapData:
        return self.ensure_output(idx)

    @property
    def op_group(self) -> str:
        return self.node_type.split("::", 1)[0]

    @property
    def op_type(self) -> str:
        parts = self.node_type.split("::", 1)
        return parts[1] if len(parts) == 2 else ""

    def get_outputs(self) -> dict[int, CapData]:
        return {idx: self.output(idx) for idx in sorted(self.outputs)}

    def to_dict(self) -> dict[str, Any]:
        info = {
            "name": self.name,
            "node_type": self.node_type,
            "config": self.config,
        }
        if self.inputs:
            info["inputs"] = {idx: f"{parent.name}:{ref}" for idx, (parent, ref) in sorted(self.inputs.items())}
        return info
