import math
from pathlib import Path

import pytest

from crg_ce.estimators.base_estimator import ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.calibration import temperature_scale_confidence
from crg_ce.estimators.openhands.config import GSNPostHocAggregateConfig
from crg_ce.estimators.openhands.oh_gsn_estimator import (
    GSNPostHocAggregateEstimator,
    aggregate_goal_confidences_with_arithmetic_mean,
    aggregate_goal_confidences_with_geometric_mean,
    aggregate_goal_confidences_with_geometric_mean_leaves,
    aggregate_goal_confidences_with_maximum,
    aggregate_goal_confidences_with_minimum,
    aggregate_goal_confidences_with_product,
    aggregate_goal_confidences_with_product_all_temperature_scaled,
    aggregate_goal_confidences_with_product_leaf_adaptive_claim_dropout,
    aggregate_goal_confidences_with_product_leaf_claim_dropout,
    aggregate_goal_confidences_with_product_leaf_temperature_scaled,
    aggregate_goal_confidences_with_temperature_scale_final,
    aggregate_goal_confidences_with_union_bound,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode


def _goal(node_id: str, confidence: float) -> GSNGoalNode:
    return GSNGoalNode(
        id=node_id,
        confidence=confidence,
        goal_name=node_id,
        auditable_claim=f"{node_id} is achieved",
        reasoning=f"{node_id} is relevant",
    )


def _evidence(node_id: str, confidence: float) -> EvidenceNodeV2:
    return EvidenceNodeV2(
        id=node_id,
        confidence=confidence,
        evidence=node_id,
        step_numbers=[1],
        auditable_claim=f"{node_id} happened",
        contribution=f"{node_id} contributes",
    )


def _ce_input(tmp_path: Path) -> ConfEstimationInput:
    archive_path = tmp_path / "conversation.tar.gz"
    archive_path.write_bytes(b"archive")
    return ConfEstimationInput(
        conversation_archive_path=archive_path,
        output_dir=tmp_path / "target-output",
        instance_id="test-instance",
        model="test-model",
        problem_statement="test problem",
    )


def test_product_aggregation_propagates_only_through_goal_children() -> None:
    # This verifies internal goals use products of child goals while evidence and evidence-backed leaf goals stay fixed.
    root = _goal("root", 0.99).model_copy(update={"confidence_rationale": "original root estimate"})
    middle = _goal("middle", 0.88)
    first_leaf = _goal("first-leaf", 0.5)
    second_leaf = _goal("second-leaf", 0.4)
    evidence = _evidence("evidence", 0.9)
    graph = ConfidenceGraph(
        nodes=[root, middle, first_leaf, second_leaf, evidence],
        edges=[
            ConfidenceEdge(source=evidence.id, target=first_leaf.id, relationship_type="supports"),
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_product(graph)
    nodes = {node.id: node for node in aggregated.nodes}

    assert nodes[evidence.id].confidence == 0.9
    assert nodes[first_leaf.id].confidence == 0.5
    assert nodes[second_leaf.id].confidence == 0.4
    assert nodes[middle.id].confidence == pytest.approx(0.2)
    assert nodes[root.id].confidence == pytest.approx(0.2)
    # This guarantees post-hoc aggregation retains the replaced rationale with an explicit method marker.
    assert nodes[root.id].confidence_rationale == "*OUTDATED: aggregated: product*: original root estimate"


def test_product_aggregation_propagates_up_a_multilevel_goal_tree() -> None:
    # This verifies products propagate bottom-up through both one-child and multi-child internal goals.
    goals = {
        "a": _goal("a", 0.1),
        "b": _goal("b", 0.2),
        "c": _goal("c", 0.3),
        "d": _goal("d", 0.9),
        "e": _goal("e", 0.8),
        "f": _goal("f", 0.5),
    }
    graph = ConfidenceGraph(
        nodes=list(goals.values()),
        edges=[
            ConfidenceEdge(source=goals["b"].id, target=goals["a"].id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=goals["c"].id, target=goals["a"].id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=goals["d"].id, target=goals["c"].id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=goals["e"].id, target=goals["c"].id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=goals["f"].id, target=goals["b"].id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=goals["a"].id,
    )

    aggregated = aggregate_goal_confidences_with_product(graph)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences["c"] == pytest.approx(0.72)
    assert confidences["b"] == pytest.approx(0.5)
    assert confidences["a"] == pytest.approx(0.36)


def test_product_aggregation_leaves_unresolvable_goal_confidences_unchanged() -> None:
    # This verifies cyclic goal dependencies cannot be resolved and retain their source-graph confidences.
    first = _goal("first", 0.7)
    second = _goal("second", 0.8)
    graph = ConfidenceGraph(
        nodes=[first, second],
        edges=[
            ConfidenceEdge(source=first.id, target=second.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second.id, target=first.id, relationship_type="decomposes_from"),
        ],
    )

    aggregated = aggregate_goal_confidences_with_product(graph)
    nodes = {node.id: node for node in aggregated.nodes}

    assert nodes[first.id].confidence == 0.7
    assert nodes[second.id].confidence == 0.8


def test_product_leaf_claim_dropout_transforms_only_original_goal_leaves() -> None:
    # This verifies the closed-form factor d + (1 - d)c is applied once at leaves and is not reapplied up the tree.
    root = _goal("root", 0.1)
    middle = _goal("middle", 0.2)
    first_leaf = _goal("first-leaf", 0.8)
    second_leaf = _goal("second-leaf", 0.2)
    graph = ConfidenceGraph(
        nodes=[root, middle, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_product_leaf_claim_dropout(
        graph,
        d=0.5,
        aggregation_type="product_leaf_claim_dropout",
    )
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences[first_leaf.id] == pytest.approx(0.9)
    assert confidences[second_leaf.id] == pytest.approx(0.6)
    assert confidences[middle.id] == pytest.approx(0.54)
    assert confidences[root.id] == pytest.approx(0.54)


def test_product_leaf_adaptive_claim_dropout_delegates_with_graph_specific_dropout() -> None:
    # This verifies k/n selects the fixed-dropout rate and k at or above n recovers the ordinary leaf product.
    root = _goal("root", 0.1)
    leaves = [_goal(f"leaf-{index}", confidence) for index, confidence in enumerate((0.8, 0.6, 0.4, 0.2))]
    graph = ConfidenceGraph(
        nodes=[root, *leaves],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="decomposes_from") for leaf in leaves],
        goal_zero_node_id=root.id,
    )

    adaptive = aggregate_goal_confidences_with_product_leaf_adaptive_claim_dropout(graph, k_expected_leaves=2)
    fixed = aggregate_goal_confidences_with_product_leaf_claim_dropout(
        graph,
        d=0.5,
        aggregation_type="product_leaf_claim_dropout",
    )
    no_dropout = aggregate_goal_confidences_with_product_leaf_adaptive_claim_dropout(graph, k_expected_leaves=4)
    adaptive_confidences = {node.id: node.confidence for node in adaptive.nodes}
    fixed_confidences = {node.id: node.confidence for node in fixed.nodes}

    assert adaptive_confidences == pytest.approx(fixed_confidences)
    assert next(node for node in no_dropout.nodes if node.id == root.id).confidence == pytest.approx(
        math.prod(leaf.confidence for leaf in leaves)
    )


def test_product_leaf_temperature_scaled_aggregates_scaled_leaf_confidences() -> None:
    # This verifies leaf-only scaling changes the leaf scores once before ordinary product propagation.
    root = _goal("root", 0.1)
    first_leaf = _goal("first-leaf", 0.8)
    second_leaf = _goal("second-leaf", 0.2)
    graph = ConfidenceGraph(
        nodes=[root, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_product_leaf_temperature_scaled(graph, temperature=2)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences[first_leaf.id] == pytest.approx(2 / 3)
    assert confidences[second_leaf.id] == pytest.approx(1 / 3)
    assert confidences[root.id] == pytest.approx(2 / 9)


def test_product_all_temperature_scaled_scales_each_goal_tree_level() -> None:
    # This verifies every derived parent product is temperature-scaled again before reaching its parent.
    root = _goal("root", 0.1)
    middle = _goal("middle", 0.2)
    first_leaf = _goal("first-leaf", 0.8)
    second_leaf = _goal("second-leaf", 0.2)
    graph = ConfidenceGraph(
        nodes=[root, middle, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_product_all_temperature_scaled(graph, temperature=2)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences[middle.id] == pytest.approx(temperature_scale_confidence(2 / 9, 2))
    assert confidences[root.id] == pytest.approx(temperature_scale_confidence(confidences[middle.id], 2))


def test_temperature_scale_final_only_changes_the_root_confidence() -> None:
    # This verifies final-only scaling preserves non-root estimates and replaces only the root rationale.
    root = _goal("root", 0.8).model_copy(update={"confidence_rationale": "root estimate"})
    leaf = _goal("leaf", 0.3)
    evidence = _evidence("evidence", 0.9)
    graph = ConfidenceGraph(
        nodes=[root, leaf, evidence],
        edges=[
            ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=evidence.id, target=leaf.id, relationship_type="supports"),
        ],
        goal_zero_node_id=root.id,
    )

    nodes = {
        node.id: node for node in aggregate_goal_confidences_with_temperature_scale_final(graph, temperature=2).nodes
    }

    assert nodes[root.id].confidence == pytest.approx(temperature_scale_confidence(0.8, 2))
    assert nodes[root.id].confidence_rationale == "*OUTDATED: aggregated: temperature_scale_final*: root estimate"
    assert nodes[leaf.id].confidence == 0.3
    assert nodes[evidence.id].confidence == 0.9


def test_temperature_scaled_products_clear_stale_unresolvable_internal_goal_confidences() -> None:
    # This guarantees both temperature-scaled aggregations retain only goal-leaf inputs, even when a malformed
    # cyclic goal structure prevents propagation; it assumes evidence confidences are not aggregation inputs.
    first = _goal("first", 0.7)
    second = _goal("second", 0.8)
    evidence = _evidence("evidence", 0.9)
    graph = ConfidenceGraph(
        nodes=[first, second, evidence],
        edges=[
            ConfidenceEdge(source=first.id, target=second.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second.id, target=first.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=evidence.id, target=first.id, relationship_type="supports"),
        ],
    )

    for aggregate in (
        aggregate_goal_confidences_with_product_leaf_temperature_scaled,
        aggregate_goal_confidences_with_product_all_temperature_scaled,
    ):
        nodes = {node.id: node for node in aggregate(graph, temperature=2).nodes}

        assert nodes[first.id].confidence == -1
        assert nodes[second.id].confidence == -1
        assert nodes[evidence.id].confidence == 0.9


def test_geometric_mean_aggregation_propagates_up_goal_tree() -> None:
    # This verifies geometric means propagate bottom-up and a one-child goal inherits its child's confidence.
    root = _goal("root", 0.1)
    middle = _goal("middle", 0.2)
    first_leaf = _goal("first-leaf", 0.9)
    second_leaf = _goal("second-leaf", 0.4)
    graph = ConfidenceGraph(
        nodes=[root, middle, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_geometric_mean(graph)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences[middle.id] == pytest.approx(0.6)
    assert confidences[root.id] == pytest.approx(0.6)


def test_geometric_mean_aggregation_avoids_intermediate_product_underflow() -> None:
    # This verifies log-space aggregation retains very small confidences whose direct product underflows to zero.
    root = _goal("root", 0.1)
    leaves = [_goal(f"leaf-{index}", 1e-100) for index in range(10)]
    graph = ConfidenceGraph(
        nodes=[root, *leaves],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="decomposes_from") for leaf in leaves],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_geometric_mean(graph)

    assert {node.id: node.confidence for node in aggregated.nodes}[root.id] == pytest.approx(1e-100)


def test_geometric_mean_leaves_aggregation_uses_only_goal_leaves() -> None:
    # This verifies leaf-only aggregation preserves leaf scores, clears interior scores, and derives the root directly.
    root = _goal("root", 0.1).model_copy(update={"confidence_rationale": "root estimate"})
    middle = _goal("middle", 0.2).model_copy(update={"confidence_rationale": "middle estimate"})
    first_leaf = _goal("first-leaf", 0.9)
    second_leaf = _goal("second-leaf", 0.4)
    graph = ConfidenceGraph(
        nodes=[root, middle, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_geometric_mean_leaves(graph)
    nodes = {node.id: node for node in aggregated.nodes}

    assert nodes[first_leaf.id].confidence == 0.9
    assert nodes[second_leaf.id].confidence == 0.4
    assert nodes[middle.id].confidence == -1
    assert nodes[root.id].confidence == pytest.approx(0.6)
    assert nodes[middle.id].confidence_rationale == "*OUTDATED: aggregated: geometric_mean_leaves*: middle estimate"
    assert nodes[root.id].confidence_rationale == "*OUTDATED: aggregated: geometric_mean_leaves*: root estimate"


def test_arithmetic_mean_aggregation_uses_mean_child_goal_confidence() -> None:
    # This verifies arithmetic means propagate through internal goals while one-child goals preserve confidence.
    root = _goal("root", 0.1)
    middle = _goal("middle", 0.2)
    first_leaf = _goal("first-leaf", 0.9)
    second_leaf = _goal("second-leaf", 0.3)
    graph = ConfidenceGraph(
        nodes=[root, middle, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="particularizes"),
        ],
        goal_zero_node_id=root.id,
    )

    aggregated = aggregate_goal_confidences_with_arithmetic_mean(graph)
    confidences = {node.id: node.confidence for node in aggregated.nodes}

    assert confidences[middle.id] == pytest.approx(0.6)
    assert confidences[root.id] == pytest.approx(0.6)


def test_minimum_and_maximum_aggregation_use_child_goal_extrema() -> None:
    # This verifies the extrema aggregations derive internal goals solely from their child-goal confidences.
    root = _goal("root", 0.1)
    first_leaf = _goal("first-leaf", 0.9)
    second_leaf = _goal("second-leaf", 0.4)
    graph = ConfidenceGraph(
        nodes=[root, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )

    minimum_graph = aggregate_goal_confidences_with_minimum(graph)
    maximum_graph = aggregate_goal_confidences_with_maximum(graph)

    assert {node.id: node.confidence for node in minimum_graph.nodes}[root.id] == 0.4
    assert {node.id: node.confidence for node in maximum_graph.nodes}[root.id] == 0.9


def test_union_bound_aggregation_propagates_summed_uncertainty_and_clamps_at_zero() -> None:
    # This verifies the Fréchet lower bound composes through the goal tree, preserves one-child particularizations,
    # and never produces a negative confidence when the children's total uncertainty exceeds one.
    root = _goal("root", 0.1).model_copy(update={"confidence_rationale": "root estimate"})
    middle = _goal("middle", 0.2)
    particularized = _goal("particularized", 0.3)
    first_leaf = _goal("first-leaf", 0.9)
    second_leaf = _goal("second-leaf", 0.8)
    third_leaf = _goal("third-leaf", 0.95)
    clamped_parent = _goal("clamped-parent", 0.6)
    low_leaf = _goal("low-leaf", 0.4)
    lower_leaf = _goal("lower-leaf", 0.3)
    graph = ConfidenceGraph(
        nodes=[
            root,
            middle,
            particularized,
            first_leaf,
            second_leaf,
            third_leaf,
            clamped_parent,
            low_leaf,
            lower_leaf,
        ],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=particularized.id, relationship_type="particularizes"),
            ConfidenceEdge(source=particularized.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=third_leaf.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=low_leaf.id, target=clamped_parent.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=lower_leaf.id, target=clamped_parent.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )

    nodes = {node.id: node for node in aggregate_goal_confidences_with_union_bound(graph).nodes}

    assert nodes[middle.id].confidence == pytest.approx(0.7)
    assert nodes[particularized.id].confidence == pytest.approx(0.7)
    assert nodes[root.id].confidence == pytest.approx(0.65)
    assert nodes[clamped_parent.id].confidence == 0.0
    assert nodes[root.id].confidence_rationale == "*OUTDATED: aggregated: union_bound*: root estimate"


def test_gsn_post_hoc_aggregate_replays_graph_and_writes_derived_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This verifies graph lookup uses replay coordinates and derived output excludes the source run's LLM usage.
    monkeypatch.chdir(tmp_path)
    source_run = Path("runs/source-gsn.yaml")
    source_run.parent.mkdir()
    source_run.write_text("estimator:\n  estimator_type: oh_gsn\n")
    source_item_dir = Path("outputs/runs/source-gsn/test-instance/test-model")
    source_item_dir.mkdir(parents=True)
    root = _goal("root", 0.99)
    first_leaf = _goal("first-leaf", 0.5)
    second_leaf = _goal("second-leaf", 0.4)
    graph = ConfidenceGraph(
        nodes=[root, first_leaf, second_leaf],
        edges=[
            ConfidenceEdge(source=first_leaf.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )
    (source_item_dir / "graph.json").write_text(graph.model_dump_json())
    (source_item_dir / "output.json").write_text(
        ConfEstimationOutput(confidence=0.99, total_tokens=100, generated_tokens=20, cost=0.01).model_dump_json()
    )
    estimator = GSNPostHocAggregateEstimator(
        GSNPostHocAggregateConfig(replay_from=source_run, aggregation_type="product")
    )

    ce_input = _ce_input(tmp_path)
    output = estimator.estimate_confidence(ce_input)

    assert output.confidence == pytest.approx(0.2)
    assert output.total_tokens == -1
    assert output.generated_tokens == -1
    assert output.cost == -1
    assert output.usage_by_model == {}
    assert ConfEstimationOutput.model_validate_json((ce_input.output_dir / "output.json").read_text()) == output
    saved_graph = ConfidenceGraph.model_validate_json((ce_input.output_dir / "graph.json").read_text())
    assert {node.id: node.confidence for node in saved_graph.nodes}[root.id] == pytest.approx(0.2)


def test_gsn_post_hoc_aggregate_rejects_missing_source_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This verifies a missing source graph fails explicitly rather than silently skipping the ablation item.
    monkeypatch.chdir(tmp_path)
    estimator = GSNPostHocAggregateEstimator(
        GSNPostHocAggregateConfig(replay_from=Path("runs/source-gsn.yaml"), aggregation_type="product")
    )

    with pytest.raises(FileNotFoundError, match="GSN graph does not exist"):
        estimator.estimate_confidence(_ce_input(tmp_path))
