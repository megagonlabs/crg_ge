import numpy as np
import pytest

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.goal_leaf_belief_propagation import (
    aggregate_goal_confidences_with_simple_bp_goal_leaf_v1,
    aggregate_goal_confidences_with_simple_bp_goal_leaf_v2,
    confidence_graph_to_simple_bp_factor_graph,
    make_decomposition_factor,
    make_leaf_confidence_factor_v2,
    make_particularization_factor,
)
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode


def _goal(node_id: str, confidence: float = -1) -> GSNGoalNode:
    return GSNGoalNode(
        id=node_id,
        confidence=confidence,
        goal_name=node_id,
        auditable_claim=f"{node_id} is true",
        reasoning="Worked-example goal.",
    )


def test_simple_bp_factors_encode_decomposition_and_particularization_truth_tables() -> None:
    # This guarantees decomposition is conjunctive and particularization is deterministic equality.
    decomposition = make_decomposition_factor("parent", ["first", "second"])
    particularization = make_particularization_factor("parent", "child")

    assert decomposition.values == pytest.approx(np.array([[[1 / 3, 1 / 3], [1 / 3, 0]], [[0, 0], [0, 1]]]))
    assert particularization.values == pytest.approx(np.array([[1, 0], [0, 1]]))


@pytest.mark.parametrize(
    ("score", "expected_emission"),
    [
        (0.0, [0.30, 0.12]),
        (0.2, [0.10, 0.05]),
        (0.4, [0.15, 0.10]),
        (0.7, [0.13, 0.08]),
        (0.9, [0.32, 0.65]),
        (0.95, [0.32, 0.65]),
        (1.0, [0.32, 0.65]),
    ],
)
def test_simple_bp_v2_leaf_factor_uses_binned_emission_probabilities(
    score: float, expected_emission: list[float]
) -> None:
    # This guarantees v2 maps scores, including exact boundaries, to the paper's [false, true] emissions.
    factor = make_leaf_confidence_factor_v2(_goal("leaf", score))

    assert factor.values == pytest.approx(expected_emission)


def test_simple_bp_v2_aggregation_uses_emission_likelihood_instead_of_raw_score() -> None:
    # This verifies the v2 aggregation path uses the selected emission row as soft evidence for inference.
    root = _goal("root").model_copy(update={"confidence_rationale": "original root estimate"})
    leaf = _goal("leaf", 0.95)
    graph = ConfidenceGraph(
        nodes=[root, leaf],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="particularizes")],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_simple_bp_goal_leaf_v2(graph)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences[root.id] == pytest.approx(0.65 / (0.32 + 0.65))
    # This guarantees BP marks a model rationale obsolete when it replaces the confidence with a marginal.
    assert {node.id: node for node in aggregated.nodes}[root.id].confidence_rationale == (
        "*OUTDATED: aggregated: simple_bp_goal_leaf_v2*: original root estimate"
    )


def test_simple_bp_matches_the_worked_goal_leaf_proof_of_concept() -> None:
    # This reproduces the proof-of-concept topology and guarantees the same root posterior from its four leaf scores.
    goals = {
        "success": _goal("success"),
        "answer": _goal("answer"),
        "correct": _goal("correct"),
        "format": _goal("format"),
        "content": _goal("content"),
        "answer-specific": _goal("answer-specific", 1.0),
        "format-specific": _goal("format-specific", 1.0),
        "kashyap": _goal("kashyap", 0.95),
        "fader": _goal("fader", 0.9),
    }
    evidence = EvidenceNodeV2(
        id="ignored-evidence",
        confidence=0.01,
        evidence="Ignored by this aggregation.",
        step_numbers=[1],
        auditable_claim="The evidence exists.",
        contribution="This v1 method deliberately ignores evidence nodes.",
    )
    graph = ConfidenceGraph(
        nodes=[*goals.values(), evidence],
        edges=[
            ConfidenceEdge(source="answer", target="success", relationship_type="decomposes_from"),
            ConfidenceEdge(source="correct", target="success", relationship_type="decomposes_from"),
            ConfidenceEdge(source="answer-specific", target="answer", relationship_type="particularizes"),
            ConfidenceEdge(source="format", target="correct", relationship_type="decomposes_from"),
            ConfidenceEdge(source="content", target="correct", relationship_type="decomposes_from"),
            ConfidenceEdge(source="format-specific", target="format", relationship_type="particularizes"),
            ConfidenceEdge(source="kashyap", target="content", relationship_type="decomposes_from"),
            ConfidenceEdge(source="fader", target="content", relationship_type="decomposes_from"),
            ConfidenceEdge(source=evidence.id, target="kashyap", relationship_type="supports"),
        ],
        goal_zero_node_id="success",
    )

    model = confidence_graph_to_simple_bp_factor_graph(graph)
    aggregated = aggregate_goal_confidences_with_simple_bp_goal_leaf_v1(graph)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert len(model.get_factors()) == 10
    assert confidences["success"] == pytest.approx(0.993758071)
    assert confidences[evidence.id] == 0.01


def test_simple_bp_rejects_missing_leaf_confidence() -> None:
    # This guarantees the method fails when a required leaf observation is absent rather than using an interior score.
    root = _goal("root", 0.99)
    leaf = _goal("leaf")
    graph = ConfidenceGraph(
        nodes=[root, leaf],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="particularizes")],
        goal_zero_node_id=root.id,
    )

    with pytest.raises(ValueError, match="has no confidence observation"):
        confidence_graph_to_simple_bp_factor_graph(graph)


def test_simple_bp_requires_goal_zero_to_be_the_only_goal_root() -> None:
    # This guarantees inference rejects disconnected goal trees instead of assigning extra implicit root priors.
    goal_zero = _goal("goal-zero", 0.8)
    disconnected_root = _goal("disconnected-root", 0.7)
    graph = ConfidenceGraph(
        nodes=[goal_zero, disconnected_root],
        edges=[],
        goal_zero_node_id=goal_zero.id,
    )

    with pytest.raises(ValueError, match="Goal graph roots must equal goal_zero_node_id"):
        confidence_graph_to_simple_bp_factor_graph(graph)
