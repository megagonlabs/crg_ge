import pytest

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode
from crg_ce.graph.utils import (
    aggregate_goal_confidences_product_interp_verbalized,
    bfs_predecessor_levels,
    bfs_predecessors,
    get_confidence_leaves,
    get_goal_leaves,
    get_max_dependent_step_number,
    get_predecessor_contexts,
    log_space_product,
)


def _goal(node_id: str, *, confidence: float = -1) -> GSNGoalNode:
    return GSNGoalNode(
        id=node_id,
        confidence=confidence,
        goal_name=node_id,
        auditable_claim=f"{node_id} is achieved",
        reasoning=f"{node_id} is necessary",
    )


def _evidence(node_id: str, step_numbers: list[int]) -> EvidenceNodeV2:
    return EvidenceNodeV2(
        id=node_id,
        evidence=node_id,
        step_numbers=step_numbers,
        auditable_claim=f"{node_id} happened",
        contribution=f"{node_id} contributes",
    )


def test_get_confidence_leaves_requires_all_incoming_sources_to_have_confidence() -> None:
    # This verifies a node is yielded only when every source node for its incoming edges has decided confidence.
    decided_source = _goal("decided-source", confidence=0.4)
    undecided_source = _goal("undecided-source")
    target = _goal("target")
    graph = ConfidenceGraph(
        nodes=[decided_source, undecided_source, target],
        edges=[
            ConfidenceEdge(source=decided_source.id, target=target.id, relationship_type="supports"),
            ConfidenceEdge(source=undecided_source.id, target=target.id, relationship_type="supports"),
        ],
    )

    confidence_leaf_ids = {node.id for node in get_confidence_leaves(graph)}

    assert target.id not in confidence_leaf_ids


def test_get_confidence_leaves_yields_node_when_all_incoming_sources_have_confidence() -> None:
    # This verifies nodes with multiple incoming edges are yielded once all source confidences are decided.
    first_source = _goal("first-source", confidence=0.4)
    second_source = _goal("second-source", confidence=0.9)
    target = _goal("target")
    graph = ConfidenceGraph(
        nodes=[first_source, second_source, target],
        edges=[
            ConfidenceEdge(source=first_source.id, target=target.id, relationship_type="supports"),
            ConfidenceEdge(source=second_source.id, target=target.id, relationship_type="supports"),
        ],
    )

    confidence_leaf_ids = {node.id for node in get_confidence_leaves(graph)}

    assert target.id in confidence_leaf_ids


def test_get_confidence_leaves_yields_node_with_no_incoming_edges() -> None:
    # This verifies nodes with no incoming edges satisfy the incoming-confidence condition vacuously.
    node = _goal("isolated")
    graph = ConfidenceGraph(nodes=[node], edges=[])

    assert list(get_confidence_leaves(graph)) == [node]


def test_get_confidence_leaves_omits_nodes_with_existing_confidence() -> None:
    # This verifies confidence leaves are only nodes whose confidence still needs to be assigned.
    node = _goal("isolated", confidence=0.5)
    graph = ConfidenceGraph(nodes=[node], edges=[])

    assert get_confidence_leaves(graph) == []


def test_get_goal_leaves_ignore_evidence_children_but_not_goal_children() -> None:
    # This verifies goal leaves are defined only by goal-to-goal ancestry; supporting evidence does not disqualify one.
    root = _goal("root")
    leaf_goal = _goal("leaf-goal")
    evidence = _evidence("evidence", [1])
    graph = ConfidenceGraph(
        nodes=[root, leaf_goal, evidence],
        edges=[
            ConfidenceEdge(source=leaf_goal.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=evidence.id, target=leaf_goal.id, relationship_type="supports"),
        ],
    )

    assert get_goal_leaves(graph) == [leaf_goal]


def test_log_space_product_handles_zero_without_taking_its_logarithm() -> None:
    # This verifies the numerically safe product preserves exact zero, which has no finite logarithm.
    assert log_space_product([0.9, 0.0, 0.8]) == 0.0


def test_product_interp_verbalized_requires_decomposition_confidence_on_every_parent() -> None:
    # This guarantees corrupted or non-interpretability graphs fail instead of silently choosing a mixing weight.
    root = _goal("root", confidence=0.8)
    leaf = _goal("leaf", confidence=0.5)
    graph = ConfidenceGraph(
        nodes=[root, leaf],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="decomposes_from")],
    )

    with pytest.raises(ValueError, match="unset confidence_in_children"):
        aggregate_goal_confidences_product_interp_verbalized(graph)


def test_get_predecessor_contexts_returns_direct_incoming_sources() -> None:
    # This verifies prompt context can be built from direct predecessors and their edge relationship labels.
    source = _goal("source", confidence=0.5)
    target = _goal("target")
    unrelated = _goal("unrelated", confidence=0.7)
    graph = ConfidenceGraph(
        nodes=[source, target, unrelated],
        edges=[ConfidenceEdge(source=source.id, target=target.id, relationship_type="supports")],
    )

    predecessor_contexts = get_predecessor_contexts(graph, target)

    assert len(predecessor_contexts) == 1
    assert predecessor_contexts[0].node == source
    assert predecessor_contexts[0].relationship_type == "supports"


def test_bfs_predecessors_yields_target_then_incoming_sources() -> None:
    # This verifies predecessor traversal starts at the target and walks incoming edge sources breadth-first.
    goal = _goal("goal")
    sub_goal = _goal("sub-goal")
    first_evidence = _evidence("first-evidence", [2])
    second_evidence = _evidence("second-evidence", [5])
    graph = ConfidenceGraph(
        nodes=[goal, sub_goal, first_evidence, second_evidence],
        edges=[
            ConfidenceEdge(source=sub_goal.id, target=goal.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=first_evidence.id, target=sub_goal.id, relationship_type="supports"),
            ConfidenceEdge(source=second_evidence.id, target=sub_goal.id, relationship_type="supports"),
        ],
    )

    visited_node_ids = [visited_node.id for visited_node in bfs_predecessors(graph, goal)]

    assert visited_node_ids == [goal.id, sub_goal.id, first_evidence.id, second_evidence.id]


def test_bfs_predecessor_levels_groups_nodes_by_distance_from_target() -> None:
    # This verifies graph renderers can consume BFS levels without reimplementing predecessor traversal.
    goal = _goal("goal")
    sub_goal = _goal("sub-goal")
    first_evidence = _evidence("first-evidence", [2])
    second_evidence = _evidence("second-evidence", [5])
    graph = ConfidenceGraph(
        nodes=[goal, sub_goal, first_evidence, second_evidence],
        edges=[
            ConfidenceEdge(source=sub_goal.id, target=goal.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=first_evidence.id, target=sub_goal.id, relationship_type="supports"),
            ConfidenceEdge(source=second_evidence.id, target=sub_goal.id, relationship_type="supports"),
        ],
    )

    levels = [[node.id for node in level] for level in bfs_predecessor_levels(graph, goal)]

    assert levels == [[goal.id], [sub_goal.id], [first_evidence.id, second_evidence.id]]


def test_get_max_dependent_step_number_uses_predecessor_bfs() -> None:
    # This verifies max step extraction considers every predecessor node with a step_numbers attribute.
    goal = _goal("goal")
    sub_goal = _goal("sub-goal")
    first_evidence = _evidence("first-evidence", [2, 4])
    second_evidence = _evidence("second-evidence", [3, 7])
    unrelated_evidence = _evidence("unrelated-evidence", [99])
    graph = ConfidenceGraph(
        nodes=[goal, sub_goal, first_evidence, second_evidence, unrelated_evidence],
        edges=[
            ConfidenceEdge(source=sub_goal.id, target=goal.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=first_evidence.id, target=sub_goal.id, relationship_type="supports"),
            ConfidenceEdge(source=second_evidence.id, target=sub_goal.id, relationship_type="supports"),
        ],
    )

    assert get_max_dependent_step_number(graph, goal) == 7
