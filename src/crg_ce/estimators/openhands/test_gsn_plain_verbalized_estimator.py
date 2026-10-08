from pathlib import Path

import pytest

from crg_ce.estimators.base_estimator import ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.openhands.config import GSNPlainVerbalizedEstimatorConfig
from crg_ce.estimators.openhands.gsn_plain_verbalized_estimator import (
    GSNPlainVerbalizedEstimator,
    render_gsn_graph,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode

ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/agronholm__typeguard.b6a7e438.combine_module__tr31kstl.tar.gz")


def _goal(node_id: str, name: str, claim: str) -> GSNGoalNode:
    return GSNGoalNode(id=node_id, goal_name=name, auditable_claim=claim, reasoning=f"Why {name} matters")


def _direct_evidence_graph() -> ConfidenceGraph:
    root = _goal("root-id", "Overall task", "The task was completed")
    child = _goal("child-id", "Implementation", "The requested implementation is correct")
    evidence = EvidenceNodeV2(
        id="evidence-id",
        evidence="Tests passed",
        step_numbers=[12, 13],
        auditable_claim="The relevant tests passed after the implementation",
        contribution="The passing tests support correctness",
    )
    return ConfidenceGraph(
        nodes=[root, child, evidence],
        edges=[
            ConfidenceEdge(source=child.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=evidence.id, target=root.id, relationship_type="supports"),
            ConfidenceEdge(source=evidence.id, target=child.id, relationship_type="proves"),
        ],
        goal_zero_node_id=root.id,
    )


def _input(output_dir: Path) -> ConfEstimationInput:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    return ConfEstimationInput(
        conversation_archive_path=ARCHIVE_PATH,
        output_dir=output_dir,
        instance_id="test-instance",
        model="trajectory-model",
        problem_statement="test-problem-statement",
    )


def test_render_gsn_graph_nests_connections_duplicates_reused_evidence_and_omits_ids() -> None:
    # This verifies the claim-centric rendering preserves every direct connection, duplicates shared evidence under each
    # affected goal, obeys the optional-field allowlist, and never exposes graph ids or stored confidence values.
    rendered = render_gsn_graph(
        _direct_evidence_graph(),
        {
            "GSNGoalNode": [],
            "EvidenceNodeV2": ["contribution"],
        },
    )

    assert rendered.startswith("ROOT GOAL\n  Name: Overall task\n  Claim: The task was completed")
    assert "- Connection: decomposes_from\n    GOAL" in rendered
    assert "- Connection: supports\n    EVIDENCE" in rendered
    assert "- Connection: proves\n        EVIDENCE" in rendered
    assert rendered.count("Name: Tests passed") == 2
    assert rendered.count("Contribution: The passing tests support correctness") == 2
    assert "Why Overall task matters" not in rendered
    assert "Step Numbers" not in rendered
    assert "root-id" not in rendered
    assert "child-id" not in rendered
    assert "evidence-id" not in rendered
    assert "Confidence" not in rendered


def test_render_gsn_graph_rejects_unreachable_nodes() -> None:
    # This verifies malformed graph content cannot be silently omitted from the prompt; it assumes graph references were
    # already structurally validated by ConfidenceGraph.
    graph = _direct_evidence_graph()
    unreachable = _goal("unreachable-id", "Unreachable", "This node is disconnected")
    graph = graph.model_copy(update={"nodes": [*graph.nodes, unreachable]})

    with pytest.raises(ValueError, match="1 nodes unreachable"):
        render_gsn_graph(graph, {})


def test_estimator_loads_graph_renders_prompt_and_parses_percentage(monkeypatch, tmp_path: Path) -> None:
    # This verifies the baseline loads the expected per-instance graph, sends only its verbalization to the configured
    # evaluator, parses its final percentage, records usage, and saves the standard confidence output.
    graph_path = tmp_path / "graphs" / "test-instance" / "trajectory-model" / "graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(_direct_evidence_graph().model_dump_json(indent=2))
    calls: list[dict] = []

    def fake_complete_text(**kwargs):
        calls.append(kwargs)
        kwargs["stats"].record_usage(
            model=kwargs["model"],
            calls=1,
            prompt_tokens=30,
            completion_tokens=5,
            reasoning_tokens=0,
            total_tokens=35,
            cost=0.02,
        )
        return "The graph contains strong supporting evidence.\nConfidence: 75%"

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.gsn_plain_verbalized_estimator.complete_text",
        fake_complete_text,
    )
    cfg = GSNPlainVerbalizedEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator"},
            "load_graphs_from_path": tmp_path / "graphs",
            "ask_and_parse_output_instruction": "prompts/confidence_estimation/litellm/"
            "ask_and_parse_output_instruction.txt",
            "graph_verbalization": {
                "node_fields": {
                    "GSNGoalNode": [],
                    "EvidenceNodeV2": ["contribution"],
                }
            },
        }
    )

    output = GSNPlainVerbalizedEstimator(cfg).estimate_confidence(_input(tmp_path / "output"))

    assert output.confidence == 0.75
    assert output.total_tokens == 35
    assert output.generated_tokens == 5
    assert output.cost == 0.02
    assert output.usage_by_model["openai/evaluator"].calls == 1
    prompt = calls[0]["messages"][0]["content"]
    assert "confidence-grounded assurance case" in prompt
    assert "ROOT GOAL" in prompt
    assert "Name: Tests passed" in prompt
    assert "root-id" not in prompt
    assert "-1" not in prompt
    assert '"Confidence: N%"' in prompt
    assert "N is an integer between 0 and 100 inclusive" in prompt
    assert ConfEstimationOutput.model_validate_json((tmp_path / "output" / "output.json").read_text()) == output
