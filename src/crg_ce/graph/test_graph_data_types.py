import json
from pathlib import Path

import pytest

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import GSNGoalNode

FIXTURE_PATH = Path(__file__).parent / "test_data" / "example_graph.json"


def test_confidence_graph_round_trip_serialization() -> None:
    raw_json = FIXTURE_PATH.read_text()

    graph_1 = ConfidenceGraph.model_validate_json(raw_json)
    serialized_json = graph_1.model_dump_json()
    graph_2 = ConfidenceGraph.model_validate_json(serialized_json)

    assert graph_2 == graph_1
    assert json.loads(serialized_json) == json.loads(raw_json)


def _node(node_id: str = "node-1") -> GSNGoalNode:
    return GSNGoalNode(
        id=node_id,
        confidence=0.75,
        goal_name="Test goal",
        auditable_claim="The test goal is satisfied",
        reasoning="Used to exercise graph validation",
    )


def test_confidence_graph_rejects_unknown_edge_source() -> None:
    with pytest.raises(ValueError, match="Unknown edge source id: missing"):
        ConfidenceGraph(
            nodes=[_node()],
            edges=[ConfidenceEdge(source="missing", target="node-1", relationship_type=None)],
        )


def test_confidence_graph_rejects_unknown_edge_target() -> None:
    with pytest.raises(ValueError, match="Unknown edge target id: missing"):
        ConfidenceGraph(
            nodes=[_node()],
            edges=[ConfidenceEdge(source="node-1", target="missing", relationship_type=None)],
        )


def test_confidence_graph_rejects_unknown_goal_zero_node_id() -> None:
    # This verifies explicit goal-zero metadata cannot reference a node outside the graph.
    with pytest.raises(ValueError, match="Unknown goal_zero_node_id: missing"):
        ConfidenceGraph(nodes=[_node()], edges=[], goal_zero_node_id="missing")
