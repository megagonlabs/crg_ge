import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from crg_ce.estimators.openhands.config import (
    AgentConfig,
    HuggingFaceDatasetConfig,
    LocalGQAMetricsConfig,
    LocalGQARunConfig,
    OutputConfig,
)
from crg_ce.evaluators.contextual_entailment import EntailmentAssessment
from crg_ce.experiments.batch_run_gqa_local import (
    LocalGQAAnalysisArtifact,
    aggregate_local_gqa_metrics,
    count_entailment_pairs,
    count_resume_work,
    main,
    parse_args,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import GSNGoalNode


def _graph() -> ConfidenceGraph:
    root = GSNGoalNode(
        id="root", goal_name="Root", auditable_claim="The task succeeds.", reasoning="Root", confidence=0.5
    )
    first = GSNGoalNode(
        id="first", goal_name="First", auditable_claim="First requirement holds.", reasoning="First", confidence=0.5
    )
    second = GSNGoalNode(
        id="second", goal_name="Second", auditable_claim="Second requirement holds.", reasoning="Second", confidence=0.5
    )
    leaf_one = GSNGoalNode(
        id="leaf-one", goal_name="Leaf one", auditable_claim="First part holds.", reasoning="Leaf", confidence=0.5
    )
    leaf_two = GSNGoalNode(
        id="leaf-two", goal_name="Leaf two", auditable_claim="Second part holds.", reasoning="Leaf", confidence=0.5
    )
    concrete = GSNGoalNode(
        id="concrete",
        goal_name="Concrete",
        auditable_claim="The concrete task succeeds.",
        reasoning="Concrete",
        confidence=0.5,
    )
    return ConfidenceGraph(
        nodes=[root, first, second, leaf_one, leaf_two, concrete],
        edges=[
            ConfidenceEdge(source="first", target="root", relationship_type="decomposes_from"),
            ConfidenceEdge(source="second", target="root", relationship_type="decomposes_from"),
            ConfidenceEdge(source="leaf-one", target="first", relationship_type="decomposes_from"),
            ConfidenceEdge(source="leaf-two", target="first", relationship_type="decomposes_from"),
            ConfidenceEdge(source="concrete", target="root", relationship_type="particularizes"),
        ],
        goal_zero_node_id="root",
    )


def test_local_gqa_counts_each_parent_decomposition_and_macro_averages(
    monkeypatch, tmp_path: Path, caplog, capsys
) -> None:
    # This verifies each decomposition parent yields one conjunction evaluation.
    # It also verifies the aggregate score is a graph macro average.
    # It assumes child-to-parent decomposes_from edges encode a jointly sufficient sibling set.
    graph_dir = tmp_path / "source-run"
    graph_path = graph_dir / "test-instance" / "test/model" / "graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(_graph().model_dump_json())
    monkeypatch.setattr(
        HuggingFaceDatasetConfig,
        "load_dataset",
        lambda self: SimpleNamespace(
            to_list=lambda: [
                {
                    "instance_id": "test-instance",
                    "model": "test/model",
                    "problem_statement": "Complete the task.",
                }
            ]
        ),
    )
    monkeypatch.setattr("crg_ce.experiments.batch_run_gqa_local.output_dir_for_run_config", lambda _: graph_dir)
    labels = iter(
        [
            "entailment",
            "neutral",
            "entailment",
            "contradiction",
            "neutral",
            "entailment",
            "neutral",
            "contradiction",
            "neutral",
            "neutral",
            "entailment",
            "neutral",
        ]
    )

    def fake_complete(**kwargs):
        return EntailmentAssessment(label=next(labels), rationale="Test assessment")  # type: ignore

    monkeypatch.setattr("crg_ce.evaluators.contextual_entailment.tenaciously_complete_structured", fake_complete)
    cfg = LocalGQARunConfig(
        agent=AgentConfig(model_name="test/evaluator"),
        graph_run_config=Path("runs/source.yaml"),
        output=OutputConfig(output_dir=tmp_path / "local-output"),
        max_workers=1,
        metrics=LocalGQAMetricsConfig(non_redundant_siblings=True, confidence_aggregation_mse=True),
    )

    main(cfg, save_prompts=True)

    output_dir = cfg.output.output_dir
    assert output_dir is not None
    artifact = LocalGQAAnalysisArtifact.model_validate_json(
        (output_dir / "test-instance" / "test/model" / "analysis.json").read_text()
    )
    assert artifact.joint_sufficiency.model_dump() == {  # type: ignore
        "score": 0.5,
        "entailment_count": 1,
        "contradiction_count": 0,
        "neutral_count": 1,
        "total_predictions": 2,
        "assessments": [
            {
                "parent_id": "root",
                "child_ids": ["first", "second"],
                "premise": "First requirement holds. AND Second requirement holds.",
                "hypothesis": "The task succeeds.",
                "assessment": {"label": "entailment", "rationale": "Test assessment"},
            },
            {
                "parent_id": "first",
                "child_ids": ["leaf-one", "leaf-two"],
                "premise": "First part holds. AND Second part holds.",
                "hypothesis": "First requirement holds.",
                "assessment": {"label": "neutral", "rationale": "Test assessment"},
            },
        ],
    }
    assert artifact.child_necessity is not None
    assert artifact.child_necessity.model_dump(exclude={"assessments"}) == {
        "score": 0.5,
        "entailment_count": 2,
        "contradiction_count": 1,
        "neutral_count": 1,
        "total_predictions": 4,
    }
    assert artifact.child_necessity.assessments[0].premise == "The task succeeds."
    assert artifact.child_necessity.assessments[0].hypothesis == "First requirement holds."
    assert artifact.non_redundant_siblings is not None
    assert artifact.non_redundant_siblings.model_dump(exclude={"assessments"}) == {
        "score": 1.0,
        "entailment_count": 0,
        "contradiction_count": 1,
        "neutral_count": 3,
        "total_predictions": 4,
    }
    assert artifact.quality_of_particularization is not None
    assert artifact.quality_of_particularization.model_dump(exclude={"assessments"}) == {
        "score": 0.0,
        "entailment_count": 1,
        "contradiction_count": 0,
        "neutral_count": 1,
        "total_predictions": 2,
    }
    assert artifact.total_entailments.model_dump() == {
        "score": 4 / 8,
        "observed_count": 4,
        "expected_count": 8,
    }
    assert artifact.confidence_aggregation_mse is not None
    assert artifact.confidence_aggregation_mse.model_dump(exclude={"comparisons", "particularization_comparisons"}) == {
        "product_mse": 0.0625,
        "geometric_mean_mse": 0.0,
        "arithmetic_mean_mse": 0.0,
        "total_decompositions": 2,
        "particularization_mse": 0.0,
        "total_particularizations": 1,
    }
    assert (output_dir / "test-instance" / "test/model" / "joint_sufficiency_root_first_second.txt").is_file()
    assert (output_dir / "test-instance" / "test/model" / "child_necessity_root_first.txt").is_file()
    assert (output_dir / "test-instance" / "test/model" / "non_redundant_siblings_root_first_second.txt").is_file()
    assert (output_dir / "test-instance" / "test/model" / "quality_of_particularization_root_concrete.txt").is_file()
    results = pd.read_csv(output_dir / "results.csv")
    assert results.loc[0, "joint_sufficiency_score"] == 0.5
    assert results.loc[0, "child_necessity_score"] == 0.5
    assert results.loc[0, "non_redundant_siblings_score"] == 1.0
    assert results.loc[0, "quality_of_particularization_score"] == 0.0
    assert results.loc[0, "total_entailments_score"] == 4 / 8
    assert results.loc[0, "total_entailments_observed_count"] == 4
    assert results.loc[0, "total_entailments_expected_count"] == 8
    assert results.loc[0, "confidence_aggregation_mse_product_mse"] == 0.0625
    aggregate = json.loads((output_dir / "metrics.json").read_text())
    assert aggregate["joint_sufficiency"]["score"] == 0.5
    assert aggregate["child_necessity"]["score"] == 0.5
    assert aggregate["non_redundant_siblings"]["score"] == 1.0
    assert aggregate["quality_of_particularization"]["score"] == 0.0
    assert aggregate["total_entailments"] == {"score": 4 / 8, "observed_count": 4, "expected_count": 8}
    assert aggregate["confidence_aggregation_mse"]["product_mse"] == 0.0625
    assert aggregate["confidence_aggregation_mse"]["particularization_mse"] == 0.0
    assert aggregate["avg_decomposition_count"] == 2.0
    assert aggregate["avg_particularization_count"] == 1.0
    score_summary = capsys.readouterr().out
    assert "joint_sufficiency: 0.5000" in score_summary
    assert "total_entailments: 0.5000 (4/8)" in score_summary
    assert "confidence_product_mse: 0.0625" in score_summary
    assert "avg_decompositions_per_graph: 2.0000" in score_summary
    assert aggregate_local_gqa_metrics([artifact, artifact]).child_necessity.score == 0.5  # type: ignore[union-attr]
    assert "Local GQA result: instance_id=test-instance model=test/model" in caplog.text
    assert 'Local GQA aggregate metrics: {"n":1,"joint_sufficiency":' in caplog.text

    # This verifies a partial redo replaces only the selected aspect in an otherwise resumed artifact.
    labels = iter(["entailment", "entailment"])
    main(cfg, redo_aspects={"quality_of_particularization"})
    redone_artifact = LocalGQAAnalysisArtifact.model_validate_json(
        (output_dir / "test-instance" / "test/model" / "analysis.json").read_text()
    )
    assert redone_artifact.joint_sufficiency == artifact.joint_sufficiency
    assert redone_artifact.non_redundant_siblings == artifact.non_redundant_siblings
    assert redone_artifact.quality_of_particularization is not None
    assert redone_artifact.quality_of_particularization.score == 1.0
    assert redone_artifact.total_entailments.score == 5 / 8


def test_dry_run_counts_enabled_sampled_pairs_without_evaluating(monkeypatch, tmp_path: Path) -> None:
    # This verifies sampling caps only sibling pairs, while CLI resume defaults on with an opt-out.
    # It guarantees all enabled decomposition and bidirectional particularization checks remain present.
    graph_dir = tmp_path / "source-run"
    graph_path = graph_dir / "test-instance" / "test/model" / "graph.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(_graph().model_dump_json())
    monkeypatch.setattr(
        HuggingFaceDatasetConfig,
        "load_dataset",
        lambda self: SimpleNamespace(
            to_list=lambda: [
                {
                    "instance_id": "test-instance",
                    "model": "test/model",
                    "problem_statement": "Complete the task.",
                }
            ]
        ),
    )
    monkeypatch.setattr("crg_ce.experiments.batch_run_gqa_local.output_dir_for_run_config", lambda _: graph_dir)
    cfg = LocalGQARunConfig(
        agent=AgentConfig(model_name="test/evaluator"),
        graph_run_config=Path("runs/source.yaml"),
        output=OutputConfig(output_dir=tmp_path / "local-output"),
        sample_n_pairs=1,
    )

    assert cfg.metrics.non_redundant_siblings is False
    assert count_entailment_pairs(cfg) == {
        "joint_sufficiency": 2,
        "child_necessity": 4,
        "non_redundant_siblings": 0,
        "quality_of_particularization": 2,
        "graphs": 1,
        "total_entailments": 8,
        "total_evaluator_pairs": 8,
    }
    assert parse_args(["config.yaml", "--dry-run"]).dry_run is True
    assert parse_args(["config.yaml"]).resume is True
    assert parse_args(["config.yaml", "--no-resume"]).resume is False
    assert parse_args(
        ["config.yaml", "--redo", "quality_of_particularization", "--redo", "non_redundant_siblings"]
    ).redo == ["quality_of_particularization", "non_redundant_siblings"]
    assert count_resume_work(cfg, resume=True, redo_aspects=set()) == {
        "resumed_graphs": 0,
        "graphs_to_analyze": 1,
        "resumed_evaluator_pairs": 0,
        "evaluator_pairs_to_run": 8,
    }
