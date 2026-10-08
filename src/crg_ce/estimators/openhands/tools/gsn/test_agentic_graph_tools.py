import pytest

from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import (
    AgenticGoalSpec,
    DivideAndConquerAction,
    DivideAndConquerInterpAction,
    DivideAndConquerInterpTool,
    GatheredEvidenceCatalogItemAgenticV1,
    GatheredEvidenceEdgeAgenticV1,
    GatherEvidenceForAgenticGraphActionV1,
    GatherEvidenceForAgenticGraphToolV1,
    ParticularizeAction,
    ParticularizeInterpAction,
    ParticularizeInterpTool,
    apply_agentic_evidence_action,
    apply_agentic_graph_action,
    configured_gather_evidence_action_type,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.nodes import GSNGoalNode
from crg_ce.utils.graphs import render_graph_claims_compact


def _goal(name: str, claim: str) -> AgenticGoalSpec:
    return AgenticGoalSpec(goal_identifier=name, auditable_claim=claim, reasoning=f"Why {name} matters")


def _root_graph() -> ConfidenceGraph:
    root = GSNGoalNode(
        goal_name="Successful Final Outcome",
        auditable_claim="The delivered result satisfies the task.",
        reasoning="Root goal",
    )
    return ConfidenceGraph(nodes=[root], edges=[], goal_zero_node_id=root.id)


def _actions() -> list[ParticularizeAction | DivideAndConquerAction]:
    return [
        ParticularizeAction(
            target_goal_identifier="Successful Final Outcome",
            target_auditable_claim="This repeated claim is not used for target resolution.",
            particularized_goal=_goal("Correct parser result", "The requested parser behavior is correct."),
        ),
        DivideAndConquerAction(
            target_goal_identifier="Correct parser result",
            target_auditable_claim="The requested parser behavior is correct.",
            sub_goals=[
                _goal("Accepted inputs", "Every required input is accepted."),
                _goal("Produced outputs", "Every required output is produced."),
            ],
        ),
    ]


def test_apply_agentic_graph_actions_preserves_heterogeneous_strategy_relationships() -> None:
    # This verifies heterogeneous strategies can update different goals sequentially within one tool-call batch.
    graph = _root_graph()
    for action in _actions():
        graph = apply_agentic_graph_action(graph, action)

    assert [edge.relationship_type for edge in graph.edges] == [
        "particularizes",
        "decomposes_from",
        "decomposes_from",
    ]
    assert render_graph_claims_compact(graph) == (
        "Node: Successful Final Outcome\n"
        "Claim: The delivered result satisfies the task.\n"
        "Parents: None\n\n"
        "Node: Correct parser result\n"
        "Claim: The requested parser behavior is correct.\n"
        "Parents: Successful Final Outcome\n\n"
        "Node: Accepted inputs\n"
        "Claim: Every required input is accepted.\n"
        "Parents: Correct parser result\n\n"
        "Node: Produced outputs\n"
        "Claim: Every required output is produced.\n"
        "Parents: Correct parser result"
    )


def test_apply_agentic_graph_action_rejects_multiple_strategies_for_one_goal() -> None:
    # This guarantees a goal cannot acquire ambiguous competing inference semantics in separate tool calls.
    graph = apply_agentic_graph_action(_root_graph(), _actions()[0])
    duplicate_strategy = ParticularizeAction(
        target_goal_identifier="Successful Final Outcome",
        target_auditable_claim="The delivered result satisfies the task.",
        particularized_goal=_goal("Other result", "Another result is correct."),
    )

    with pytest.raises(ValueError, match="more than one strategy"):
        apply_agentic_graph_action(graph, duplicate_strategy)


def test_divide_and_conquer_requires_at_least_two_children() -> None:
    # This verifies the structured tool reserves divide-and-conquer for genuine multi-claim conjunctions.
    with pytest.raises(ValueError, match="at least 2"):
        DivideAndConquerAction(
            target_goal_identifier="Goal",
            target_auditable_claim="The goal holds.",
            sub_goals=[_goal("Only child", "The only child holds.")],
        )


@pytest.mark.parametrize(
    ("tool", "action_type"),
    [
        (DivideAndConquerInterpTool, DivideAndConquerInterpAction),
        (ParticularizeInterpTool, ParticularizeInterpAction),
    ],
)
def test_interp_graph_tools_require_confidence(tool, action_type) -> None:
    # This guarantees both interpretability variants expose confidence as a required tool-call argument.
    definition = tool.create()[0]

    assert definition.action_type is action_type
    assert "confidence" in action_type.model_json_schema()["required"]


def test_interp_graph_action_records_confidence_on_parent_goal() -> None:
    # This verifies interpretability calls persist confidence in the parent-to-children relationship on the graph.
    action = ParticularizeInterpAction(
        target_goal_identifier="Successful Final Outcome",
        target_auditable_claim="The delivered result satisfies the task.",
        particularized_goal=_goal("Concrete outcome", "The concrete outcome satisfies the task."),
        confidence=0.85,
    )

    graph = apply_agentic_graph_action(_root_graph(), action)

    assert graph.nodes[0].confidence_in_children == 0.85  # type: ignore[attr-defined]
    assert graph.nodes[1].confidence_in_children == -1  # type: ignore[attr-defined]


def test_apply_agentic_evidence_action_matches_v3_catalog_and_identifier_edges() -> None:
    # This verifies the agentic V1 tool preserves LiteLLM V3 evidence semantics while targeting goal identifiers.
    graph = apply_agentic_graph_action(_root_graph(), _actions()[0])
    action = GatherEvidenceForAgenticGraphActionV1(
        evidence_catalog=[
            GatheredEvidenceCatalogItemAgenticV1(
                evidence_key="implementation",
                evidence="The parser implementation returns the requested result.",
                step_numbers=[4, 7],
                auditable_claim="The implementation produces the requested parser result.",
                contribution="Directly supports the concrete result claim.",
            )
        ],
        evidence_edges=[
            GatheredEvidenceEdgeAgenticV1(
                evidence_key="implementation",
                target_goal_identifier="Correct parser result",
                relationship_type="supports",
            )
        ],
    )

    updated_graph = apply_agentic_evidence_action(graph, action)

    evidence_node = updated_graph.nodes[-1]
    assert evidence_node.auditable_claim == "The implementation produces the requested parser result."  # type: ignore
    assert updated_graph.edges[-1].source == evidence_node.id
    assert updated_graph.edges[-1].target == graph.nodes[-1].id
    assert updated_graph.edges[-1].relationship_type == "supports"


def test_configured_evidence_tool_schema_rejects_unconfigured_edge_labels() -> None:
    # This guarantees the tool schema and Pydantic parser admit only the evidence labels selected for an agentic run.
    tool = GatherEvidenceForAgenticGraphToolV1.create(evidence_edge_labels=["supports", "undermines"])[0]
    action_type = tool.action_type
    schema = action_type.model_json_schema()
    edge_schema = next(
        definition
        for name, definition in schema["$defs"].items()
        if name.startswith("ConfiguredGatheredEvidenceEdgeAgenticV1_")
    )
    assert edge_schema["properties"]["relationship_type"]["enum"] == ["supports", "undermines"]

    evidence_catalog = [
        {
            "evidence_key": "evidence",
            "evidence": "The evidence.",
            "step_numbers": [1],
            "auditable_claim": "The evidence holds.",
            "contribution": "It is relevant.",
        }
    ]
    action_type.model_validate(
        {
            "evidence_catalog": evidence_catalog,
            "evidence_edges": [
                {
                    "evidence_key": "evidence",
                    "target_goal_identifier": "Goal",
                    "relationship_type": "supports",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="Input should be 'supports' or 'undermines'"):
        action_type.model_validate(
            {
                "evidence_catalog": evidence_catalog,
                "evidence_edges": [
                    {
                        "evidence_key": "evidence",
                        "target_goal_identifier": "Goal",
                        "relationship_type": "proves",
                    }
                ],
            }
        )


def test_configured_evidence_action_type_reuses_the_same_dynamic_class() -> None:
    # This guarantees repeated agentic runs do not register distinct same-named Pydantic action classes.
    first = configured_gather_evidence_action_type(["supports", "undermines", "unverified"])
    second = configured_gather_evidence_action_type(("supports", "undermines", "unverified"))

    assert first is second


def test_apply_agentic_evidence_action_rejects_non_leaf_goal_identifier() -> None:
    # This verifies V1 evidence can target only the graph's current atomic leaf goals.
    action = GatherEvidenceForAgenticGraphActionV1(
        evidence_catalog=[
            GatheredEvidenceCatalogItemAgenticV1(
                evidence_key="evidence",
                evidence="Evidence",
                step_numbers=[1],
                auditable_claim="Evidence holds.",
                contribution="Contribution.",
            )
        ],
        evidence_edges=[
            GatheredEvidenceEdgeAgenticV1(
                evidence_key="evidence",
                target_goal_identifier="Successful Final Outcome",
                relationship_type="supports",
            )
        ],
    )
    graph = apply_agentic_graph_action(_root_graph(), _actions()[0])

    with pytest.raises(ValueError, match="Unknown target goal identifiers"):
        apply_agentic_evidence_action(graph, action)
