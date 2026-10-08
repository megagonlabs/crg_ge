from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import GSNGoalNode
from crg_ce.utils.graphs import get_node_depths, max_node_depth, render_graph_claims_compact


def test_max_node_depth_uses_shortest_root_distance() -> None:
    graph = {
        "goal_zero_node_id": "root",
        "nodes": [
            {"id": "root", "kind": "GSNGoalNode"},
            {"id": "near", "kind": "GSNGoalNode"},
            {"id": "far", "kind": "GSNGoalNode"},
            {"id": "shared", "kind": "GSNGoalNode"},
        ],
        "edges": [
            {"source": "near", "target": "root", "relationship_type": "decomposes_from"},
            {"source": "far", "target": "near", "relationship_type": "decomposes_from"},
            {"source": "shared", "target": "far", "relationship_type": "decomposes_from"},
            {"source": "shared", "target": "root", "relationship_type": "decomposes_from"},
        ],
    }

    assert get_node_depths(graph) == {"root": 0, "near": 1, "shared": 1, "far": 2}
    assert max_node_depth(graph) == 2


def test_render_graph_claims_compact_numbers_nodes_in_breadth_first_order() -> None:
    root = GSNGoalNode(
        id="root",
        goal_name="Root",
        auditable_claim="The task succeeds.",
        reasoning="Root goal",
    )
    abstract = GSNGoalNode(
        id="abstract",
        goal_name="Abstract",
        auditable_claim="The result is correct.",
        reasoning="Abstract condition",
    )
    concrete = GSNGoalNode(
        id="concrete",
        goal_name="Concrete",
        auditable_claim="The requested parser is correct.",
        reasoning="Concrete condition",
    )
    graph = ConfidenceGraph(
        nodes=[root, abstract, concrete],
        edges=[
            ConfidenceEdge(source=abstract.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=concrete.id, target=abstract.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    assert render_graph_claims_compact(graph) == (
        "Node: Root\n"
        "Claim: The task succeeds.\n"
        "Parents: None\n\n"
        "Node: Abstract\n"
        "Claim: The result is correct.\n"
        "Parents: Root\n\n"
        "Node: Concrete\n"
        "Claim: The requested parser is correct.\n"
        "Parents: Abstract"
    )
