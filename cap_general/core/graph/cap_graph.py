"""CapGraph: a directed acyclic graph of CapNodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cap_general.core.graph.cap_node import CapNode


class CapGraph:
    """DAG of CapNodes. Nodes are executed in topological order."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._nodes: dict[str, CapNode] = {}
        self._node_names: list[str] = []

    def __str__(self) -> str:
        return (
            f"GRAPH({self.name},nodes_num={len(self._node_names)})"
            + "\n\nNODES:\n"
            + "\n\n".join(str(self._nodes[n]) for n in self._node_names)
        )

    def add_node(
        self,
        name: str,
        node_type: str,
        config: dict[str, Any] | None = None,
        inputs: dict[int, Any] | list[Any] | None = None,
    ) -> CapNode:
        """Add a node and wire parent -> child edges from ``inputs``."""
        if name in self._nodes:
            raise ValueError(f"Duplicate node name in CapGraph {self.name!r}: {name}")
        node = CapNode(name=name, node_type=node_type, config=config or {})
        self._nodes[name] = node
        self._node_names.append(name)
        if inputs:
            input_items = inputs.items() if isinstance(inputs, dict) else enumerate(inputs)
            for input_idx, input_value in input_items:
                input_idx = int(input_idx)
                if isinstance(input_value, CapNode):
                    parent, output_idx = input_value, 0
                elif isinstance(input_value, tuple) and len(input_value) == 2:
                    parent, output_idx = input_value
                    if not isinstance(parent, CapNode):
                        parent = self._nodes[parent]
                    output_idx = int(output_idx)
                elif isinstance(input_value, str):
                    if ":" in input_value:
                        parent_name, output_idx = input_value.split(":", 1)
                        parent = self._nodes[parent_name]
                        output_idx = int(output_idx)
                    else:
                        parent, output_idx = self._nodes[input_value], 0
                else:
                    raise TypeError(f"Unexpected graph input value: {input_value!r}")
                node.add_input(input_idx, parent, output_idx)
        return node

    def get_node(self, name: str) -> CapNode:
        return self._nodes[name]

    def find_data(self, name: str):
        node_name, ref = name.split(":", 1)
        return self.get_node(node_name).output(int(ref))

    def nodes(self) -> list[CapNode]:
        """Return all nodes in topological order (sources first)."""
        if not self._node_names:
            return []
        return self._topological_sort()

    def _topological_sort(self) -> list[CapNode]:
        in_degree = {name: len(node.parents) for name, node in self._nodes.items()}
        node_order = {name: idx for idx, name in enumerate(self._node_names)}
        queue = [self._nodes[name] for name in self._node_names if in_degree[name] == 0]
        result: list[CapNode] = []
        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in sorted(node.children, key=lambda n: node_order[n.name]):
                in_degree[child.name] -= 1
                if in_degree[child.name] == 0:
                    queue.append(child)
        if len(result) != len(self._nodes):
            raise ValueError(f"CapGraph '{self.name}' contains a cycle")
        return result

    def visualize(self, directory: str | Path, name: str | None = None, fmt: str = "png") -> str:
        """Render the graph with graphviz and save to *directory*.

        Falls back to saving a .dot source file when the dot executable is absent.
        """
        try:
            import graphviz
        except ImportError as exc:
            raise ImportError("pip install graphviz to use CapGraph.visualize()") from exc

        graph_name = name or self.name
        dot = graphviz.Digraph(
            graph_name,
            node_attr={"shape": "box", "fontname": "Helvetica,Arial,sans-serif", "style": "filled"},
            edge_attr={"fontname": "Helvetica,Arial,sans-serif"},
        )
        for node in self.nodes():
            label = f"{node.op_group}::{node.op_type}\n[{node.name}]"
            if node.config:
                cfg_lines = "\n".join(f"{k}: {v}" for k, v in list(node.config.items())[:8])
                label += f"\n---\n{cfg_lines}"
            if node.inputs:
                input_lines = "\n".join(
                    f"{idx} <- {parent.name}:{ref}" for idx, (parent, ref) in sorted(node.inputs.items())
                )
                label += f"\n---\n{input_lines}"
            dot.node(node.name, label)
        for node in self.nodes():
            if node.inputs:
                for input_idx, (parent, output_idx) in sorted(node.inputs.items()):
                    dot.edge(parent.name, node.name, label=f"{input_idx}:{output_idx}")
            else:
                for parent in node.parents:
                    dot.edge(parent.name, node.name)

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            return dot.render(filename=graph_name, directory=str(out_dir), format=fmt, cleanup=True)
        except Exception as exc:
            if type(exc).__name__ != "ExecutableNotFound":
                raise
            dot_path = out_dir / f"{graph_name}.dot"
            dot_path.write_text(dot.source, encoding="utf-8")
            return str(dot_path)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "nodes": [node.to_dict() for node in self.nodes()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CapGraph":
        graph = cls(data["name"])
        for node_data in data["nodes"]:
            graph.add_node(
                name=node_data["name"],
                node_type=node_data.get("node_type") or f"{node_data['op_group']}::{node_data['op_type']}",
                config=node_data.get("config", {}),
                inputs=node_data.get("inputs"),
            )
        return graph
