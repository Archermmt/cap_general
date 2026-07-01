import logging

import pytest

from cap_general.core.graph import CapData, CapGraph
from cap_general.core.operator import BaseOperator, to_stage_fn
from cap_general.core.policy import BasePolicy

LOGGER = logging.getLogger(__name__)


@BaseOperator.register()
class _TestSourceOp(BaseOperator):
    op_group = "test"
    op_type = "source"

    @to_stage_fn
    def inference(self, inputs):
        return {"output": inputs.get("seed")}


@BaseOperator.register()
class _TestMidOp(BaseOperator):
    op_group = "test"
    op_type = "mid"

    @to_stage_fn
    def inference(self, inputs):
        return {"output": inputs.get("output")}


@BaseOperator.register()
class _TestSinkOp(BaseOperator):
    op_group = "test"
    op_type = "sink"

    @to_stage_fn
    def inference(self, inputs):
        return {"output": inputs.get("obs")}


def test_operator_run_only_allows_stage_funcs():
    operator = _TestSourceOp(config={}, logger=LOGGER)
    operator.reset()
    operator.set_stage("reset")

    with pytest.raises(AttributeError, match="has no stage 'reset'"):
        operator.run({})


def test_cap_data_str_and_dict_roundtrip():
    data = CapData(name="node:0").set("x", 1).set("y", 2)

    assert "node:0" in str(data)
    assert "x=1" in str(data)
    assert data.to_dict() == {"x": 1, "y": 2}
    assert CapData.from_dict({"x": 1}, name="foo").name == "foo"


def test_cap_graph_add_node_uses_node_objects_for_edges():
    graph = CapGraph("demo")
    a = graph.add_node("a", node_type="test::source", config={})
    b = graph.add_node("b", node_type="test::mid", config={}, inputs={0: "a:0"})
    c = graph.add_node("c", node_type="test::sink", config={}, inputs={0: "a:0", 1: b})

    assert b.parents == [a]
    assert a.children == [b, c]
    assert c.parents == [a, b]
    assert c.inputs[0] == (a, 0)
    assert c.inputs[1] == (b, 0)
    assert [node.name for node in graph.nodes()] == ["a", "b", "c"]


def test_cap_graph_to_dict_from_dict_roundtrip_with_inputs():
    graph = CapGraph("demo")
    graph.add_node("a", node_type="test::source", config={"x": 1})
    graph.add_node("b", node_type="test::sink", config={}, inputs={0: "a:0"})

    dumped = graph.to_dict()
    loaded = CapGraph.from_dict(dumped)
    a = loaded.get_node("a")
    b = loaded.get_node("b")

    assert dumped["nodes"][1]["inputs"] == {0: "a:0"}
    assert b.parents == [a]
    assert a.children == [b]
    assert b.input(0).name == "a:0"


def test_cap_graph_and_node_str_include_structure_details():
    graph = CapGraph("demo")
    a = graph.add_node("a", node_type="test::source", config={"x": 1})
    b = graph.add_node("b", node_type="test::sink", config={}, inputs={0: a})

    assert "GRAPH(demo" in str(graph)
    assert "a(test/source)" in str(a)
    assert "P:a" in str(b)
    assert "IN: 0->a:0" in str(b)


@BasePolicy.register()
class _GraphPolicy(BasePolicy):
    name = "Graph Policy"

    policy_type = "graph_policy_test"


def test_base_policy_from_config_runs_graph_roundtrip():
    policy = _GraphPolicy.from_config(
        {
            "type": "graph_policy_test",
            "name": "graph_test",
            "graph": {
                "name": "graph_test",
                "nodes": [
                    {"name": "a", "node_type": "test::source", "config": {}},
                    {
                        "name": "b",
                        "node_type": "test::sink",
                        "config": {},
                        "inputs": {0: "a:0"},
                    },
                ],
            },
        },
        logger=LOGGER,
    )
    policy.reset()

    assert policy._graph.name == "graph_test"
    assert [node.name for node in policy._graph.nodes()] == ["a", "b"]
    assert "GRAPH(graph_test" in str(policy)
