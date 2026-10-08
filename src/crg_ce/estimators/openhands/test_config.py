from pathlib import Path

import pytest
import yaml
from datasets import Dataset
from pydantic import Field, TypeAdapter, ValidationError

from crg_ce.estimators.openhands.config import (
    AgentConfig,
    BaseRunConfig,
    BasicLiteLLMVerbalEstimatorConfig,
    GSNPlainVerbalizedEstimatorConfig,
    GSNPostHocAggregateConfig,
    GSNPromptConfig,
    HuggingFaceDatasetConfig,
    OHGSNEstimatorConfig,
    OutputConfig,
    SupportedConfidenceEstimatorConfig,
    load_run_config,
)
from crg_ce.estimators.openhands.replay_estimator import ReplayEstimatorConfig
from crg_ce.resources import read_resource
from crg_ce.utils.general import resolve_template


class DummyRunConfig(BaseRunConfig):
    name: str
    output: OutputConfig = Field(default_factory=OutputConfig)


@pytest.mark.parametrize(
    "tools_preset",
    [
        "confidence_estimation",
        "node_generation",
        "gsn_goal_decomposition",
        "gsn_gather_evidence",
        "gsn_gather_evidence_v2",
        "openhands_default",
    ],
)
def test_agent_config_rejects_removed_tool_presets(tools_preset: str) -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"tools_preset": tools_preset})


def test_load_config_uses_sys_argv_and_sets_default_output_dir(tmp_path: Path, monkeypatch) -> None:
    # This test verifies that load_config reads sys.argv[1], validates using the requested run config type,
    # and infers outputs/<config path without suffix> when output.output_dir is omitted.
    config_path = tmp_path / "config" / "runs" / "example.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("name: demo\n")
    monkeypatch.chdir(tmp_path)

    cfg = load_run_config("config/runs/example.yaml", DummyRunConfig)

    assert isinstance(cfg, DummyRunConfig)
    assert cfg.name == "demo"
    assert cfg.output.output_dir == Path("outputs/config/runs/example")
    assert (cfg.output.output_dir / "run_config.yaml").exists()
    assert not (cfg.output.output_dir / "instantiated_run_config.yaml").exists()


def test_load_config_preserves_explicit_output_dir(tmp_path: Path, monkeypatch) -> None:
    # This test verifies that an explicit output.output_dir from the config file is not overwritten.
    config_path = tmp_path / "run.yaml"
    config_path.write_text("name: demo\noutput:\n  output_dir: custom/out\n")
    monkeypatch.chdir(tmp_path)

    cfg = load_run_config(
        "run.yaml",
        DummyRunConfig,
    )

    assert cfg.output.output_dir == Path("custom/out")


def test_huggingface_dataset_config_shuffles_and_samples_deterministically(monkeypatch) -> None:
    # This verifies a seeded shuffle followed by sampling selects the same random subset on every run.
    dataset = Dataset.from_dict({"instance_id": [str(index) for index in range(20)]})
    monkeypatch.setattr("crg_ce.estimators.openhands.config.load_dataset", lambda **kwargs: dataset)
    config = HuggingFaceDatasetConfig(shuffle=True, sample=10, seed=42)

    first_sample = config.load_dataset()["instance_id"]
    second_sample = config.load_dataset()["instance_id"]

    assert first_sample == second_sample
    assert len(first_sample) == 10
    assert first_sample != [str(index) for index in range(10)]


def test_supported_confidence_estimator_config_parses_replay_estimator() -> None:
    # This verifies replay source runs can be selected through the discriminated estimator configuration.
    cfg: SupportedConfidenceEstimatorConfig = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
        {
            "estimator_type": "replay",
            "replay_from": "runs/source-run.yaml",
        }
    )

    assert isinstance(cfg, ReplayEstimatorConfig)
    assert cfg.replay_from == Path("runs/source-run.yaml")


def test_supported_confidence_estimator_config_parses_gsn_post_hoc_aggregate() -> None:
    # This verifies product aggregation requires an explicit method and participates in the estimator union.
    cfg: SupportedConfidenceEstimatorConfig = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
        {
            "estimator_type": "gsn_aggregate",
            "replay_from": "runs/source-gsn.yaml",
            "aggregation_type": "product",
        }
    )

    assert isinstance(cfg, GSNPostHocAggregateConfig)
    assert cfg.aggregation_type == "product"

    geometric_mean_cfg = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(  # type: ignore
        {
            "estimator_type": "gsn_aggregate",
            "replay_from": "runs/source-gsn.yaml",
            "aggregation_type": "geometric_mean",
        }
    )
    assert isinstance(geometric_mean_cfg, GSNPostHocAggregateConfig)
    assert geometric_mean_cfg.aggregation_type == "geometric_mean"

    for aggregation_type in (
        "arithmetic_mean",
        "geometric_mean_leaves",
        "minimum",
        "maximum",
        "union_bound",
        "simple_bp_goal_leaf_v1",
        "simple_bp_goal_leaf_v2",
    ):
        additional_aggregate_cfg = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(  # type: ignore
            {
                "estimator_type": "gsn_aggregate",
                "replay_from": "runs/source-gsn.yaml",
                "aggregation_type": aggregation_type,
            }
        )
        assert isinstance(additional_aggregate_cfg, GSNPostHocAggregateConfig)
        assert additional_aggregate_cfg.aggregation_type == aggregation_type

    # This verifies prior interpolation accepts its required fixed prior for replayed child confidences.
    prior_interp_cfg = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(  # type: ignore
        {
            "estimator_type": "gsn_aggregate",
            "replay_from": "runs/source-gsn.yaml",
            "aggregation_type": "product_interp_prior",
            "interpolation_prior": 0.5,
        }
    )
    assert isinstance(prior_interp_cfg, GSNPostHocAggregateConfig)
    assert prior_interp_cfg.interpolation_prior == 0.5

    # This verifies claim dropout accepts its required probability and rejects omission of that parameter.
    claim_dropout_cfg = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(  # type: ignore
        {
            "estimator_type": "gsn_aggregate",
            "replay_from": "runs/source-gsn.yaml",
            "aggregation_type": "product_leaf_claim_dropout",
            "d": 0.25,
        }
    )
    assert isinstance(claim_dropout_cfg, GSNPostHocAggregateConfig)
    assert claim_dropout_cfg.d == 0.25

    # This verifies adaptive claim dropout accepts its required expected-leaf count.
    adaptive_claim_dropout_cfg = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(  # type: ignore
        {
            "estimator_type": "gsn_aggregate",
            "replay_from": "runs/source-gsn.yaml",
            "aggregation_type": "product_leaf_adaptive_claim_dropout",
            "k_expected_leaves": 4,
        }
    )
    assert isinstance(adaptive_claim_dropout_cfg, GSNPostHocAggregateConfig)
    assert adaptive_claim_dropout_cfg.k_expected_leaves == 4

    with pytest.raises(ValidationError):
        TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
            {
                "estimator_type": "gsn_aggregate",
                "replay_from": "runs/source-gsn.yaml",
                "aggregation_type": "product_leaf_adaptive_claim_dropout",
            }
        )

    with pytest.raises(ValidationError):
        TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
            {
                "estimator_type": "gsn_aggregate",
                "replay_from": "runs/source-gsn.yaml",
                "aggregation_type": "product_leaf_claim_dropout",
            }
        )

    for aggregation_type in (
        "product_leaf_temperature_scaled",
        "product_all_temperature_scaled",
        "temperature_scale_final",
    ):
        temperature_scaled_cfg = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(  # type: ignore
            {
                "estimator_type": "gsn_aggregate",
                "replay_from": "runs/source-gsn.yaml",
                "aggregation_type": aggregation_type,
                "temperature": 2.0,
            }
        )
        assert isinstance(temperature_scaled_cfg, GSNPostHocAggregateConfig)
        assert temperature_scaled_cfg.temperature == 2.0

    with pytest.raises(ValidationError):
        TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
            {"estimator_type": "gsn_aggregate", "replay_from": "runs/source-gsn.yaml"}
        )


def test_supported_confidence_estimator_config_parses_explicit_oh_gsn() -> None:
    # This verifies the batch estimator union accepts the retained constructor/populator pipeline.
    cfg: SupportedConfidenceEstimatorConfig = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
        {
            "estimator_type": "oh_gsn",
            "graph_generator": {
                "generator_type": "agentic",
                "agent": {
                    "model_name": "openai/test-model",
                    "tools_preset": "gsn_agentic_graph_construction",
                },
                "prompts": {
                    "gather_evidence_method": "all_at_once",
                    "goal_decompose_instruction": "prompts/gsn/direct/agentic/goal_decomposition.j2",
                    "gather_evidence_instruction": "prompts/gsn/direct/agentic/evidence_gathering.j2",
                    "assign_confidence_to_goal_instruction": "prompts/not_used.j2",
                    "assign_confidence_to_evidence_instruction": "prompts/not_used.j2",
                },
            },
            "graph_populator": {
                "generator_type": "litellm",
                "agent": {"model_name": "openai/test-model", "tools_preset": "default"},
                "prompts": {
                    "gather_evidence_method": "all_at_once",
                    "goal_decompose_instruction": "prompts/not_used.j2",
                    "gather_evidence_instruction": "prompts/not_used.j2",
                    "assign_confidence_to_goal_instruction": (
                        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
                    ),
                    "assign_confidence_to_evidence_instruction": (
                        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_evidence.j2"
                    ),
                },
                "verbalization": {"scale_min": 0, "scale_max": 10},
            },
        }
    )

    assert isinstance(cfg, OHGSNEstimatorConfig)
    assert cfg.graph_generator.generator_type == "agentic"
    assert cfg.graph_generator.prompts.gather_evidence_method == "all_at_once"
    assert cfg.graph_populator.generator_type == "litellm"


def test_supported_confidence_estimator_config_parses_litellm_verbal() -> None:
    # This verifies ex-situ verbal estimator configs select a direct query mode and evaluator model.
    cfg: SupportedConfidenceEstimatorConfig = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
        {
            "estimator_type": "litellm_verbal",
            "litellm": {
                "agent": {"model_name": "openai/test-model"},
                "query_type": "ask_and_parse",
            },
        }
    )

    assert isinstance(cfg, BasicLiteLLMVerbalEstimatorConfig)
    assert cfg.litellm.agent.model_name == "openai/test-model"
    assert cfg.litellm.query_type == "ask_and_parse"


def test_supported_confidence_estimator_config_parses_gsn_plain_verbalized_fields() -> None:
    # This verifies the tagged baseline config accepts per-node optional-field allowlists while preserving mandatory
    # graph-loading and evaluator settings.
    cfg: SupportedConfidenceEstimatorConfig = TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python(
        {
            "estimator_type": "gsn_plain_verbalized",
            "agent": {"model_name": "openai/evaluator"},
            "load_graphs_from_path": "outputs/graphs",
            "graph_verbalization": {
                "node_fields": {
                    "GSNGoalNode": [],
                    "EvidenceNodeV2": ["contribution"],
                }
            },
        }
    )

    assert isinstance(cfg, GSNPlainVerbalizedEstimatorConfig)
    assert cfg.load_graphs_from_path == Path("outputs/graphs")
    assert cfg.graph_verbalization.node_fields == {
        "GSNGoalNode": [],
        "EvidenceNodeV2": ["contribution"],
    }


def test_gsn_plain_verbalized_config_rejects_identifier_and_core_fields() -> None:
    # This verifies ids can never be configured into prompts and core name/claim fields cannot be redundantly toggled.
    with pytest.raises(ValidationError, match="Unsupported fields for EvidenceNodeV2"):
        GSNPlainVerbalizedEstimatorConfig.model_validate(
            {
                "load_graphs_from_path": "graphs",
                "graph_verbalization": {"node_fields": {"EvidenceNodeV2": ["id"]}},
            }
        )


def test_gsn_prompt_config_requires_problem_statement_when_skipping_initial_trajectory_messages() -> None:
    # This verifies removing the initial task messages requires the phase-one prompt to receive the task separately.
    prompt_config = {
        "goal_decompose_instruction": "prompts/not_used.j2",
        "gather_evidence_instruction": "prompts/not_used.j2",
        "assign_confidence_to_goal_instruction": "prompts/not_used.j2",
        "assign_confidence_to_evidence_instruction": "prompts/not_used.j2",
        "skip_system_user_messages_in_trajectory": True,
    }

    with pytest.raises(ValidationError, match="requires problem_statement"):
        GSNPromptConfig.model_validate(prompt_config)

    configured = GSNPromptConfig.model_validate(
        {**prompt_config, "decompose_goal_included_fields": ["trajectory", "problem_statement"]}
    )

    assert configured.skip_system_user_messages_in_trajectory is True

    with pytest.raises(ValidationError, match="Core fields are always rendered for GSNGoalNode"):
        GSNPlainVerbalizedEstimatorConfig.model_validate(
            {
                "load_graphs_from_path": "graphs",
                "graph_verbalization": {"node_fields": {"GSNGoalNode": ["goal_name"]}},
            }
        )


def test_interp_population_requires_interp_construction() -> None:
    # This guarantees interpolation cannot run without construction confidences while allowing confidence-aware
    # construction to feed population modes that do not use those additional values.
    run_config = yaml.safe_load(
        Path("runs/ablations/gc_max_depth/qwen38_27b_agentic_fp8_max_depth_4.yaml").read_text()
    )
    estimator = run_config["estimator"]
    estimator["graph_generator"]["agent"]["tools_preset"] = "gsn_agentic_graph_construction_interp"
    estimator["graph_populator"]["confidence_population_mode"] = "product_interp_verbalized"

    configured = OHGSNEstimatorConfig.model_validate(estimator)

    assert configured.graph_generator.agent.tools_preset == "gsn_agentic_graph_construction_interp"
    assert configured.graph_populator.confidence_population_mode == "product_interp_verbalized"
    assert configured.graph_populator.interpolation_prior == 0.5

    estimator["graph_generator"]["agent"]["tools_preset"] = "default"
    with pytest.raises(ValidationError, match="require gsn_agentic_graph_construction_interp"):
        OHGSNEstimatorConfig.model_validate(estimator)

    estimator["graph_generator"]["agent"]["tools_preset"] = "gsn_agentic_graph_construction_interp"
    estimator["graph_populator"]["confidence_population_mode"] = "goal_leaves_product"
    OHGSNEstimatorConfig.model_validate(estimator)

    estimator["graph_populator"]["confidence_population_mode"] = "product_interp_prior"
    estimator["graph_populator"]["interpolation_prior"] = 0.2
    configured_with_prior = OHGSNEstimatorConfig.model_validate(estimator)
    assert configured_with_prior.graph_populator.interpolation_prior == 0.2


def test_gsn_prompt_config_defaults_domain_success_criteria_to_swe() -> None:
    # This verifies system prompts receive SWE criteria unless a run config overrides or disables the domain criteria.
    prompt_config = GSNPromptConfig.model_validate(
        {
            "goal_decompose_instruction": "prompts/not_used.j2",
            "gather_evidence_instruction": "prompts/not_used.j2",
            "assign_confidence_to_goal_instruction": "prompts/not_used.j2",
            "assign_confidence_to_evidence_instruction": "prompts/not_used.j2",
        }
    )

    assert (
        prompt_config.render_domain_success_criteria()
        == read_resource("prompts/domains/swe/agent_success_criteria.txt").strip()
    )
    assert prompt_config.evidence_edge_labels == ["proves", "supports", "refutes", "undermines"]
    prompt_config.domain_success_criteria = None
    assert prompt_config.render_domain_success_criteria() is None


def test_gsn_prompt_config_selects_domain_success_criteria_by_benchmark() -> None:
    # This verifies benchmark-specific criteria resolve from YAML template paths and unknown benchmarks fail explicitly.
    prompt_config = GSNPromptConfig.model_validate(
        {
            "domain_success_criteria": {
                "by_benchmark": {"swe-bench": "prompts/domains/swe/agent_success_criteria.txt"}
            },
            "goal_decompose_instruction": "prompts/not_used.j2",
            "gather_evidence_instruction": "prompts/not_used.j2",
            "assign_confidence_to_goal_instruction": "prompts/not_used.j2",
            "assign_confidence_to_evidence_instruction": "prompts/not_used.j2",
        }
    )

    assert (
        prompt_config.render_domain_success_criteria("swe-bench")
        == read_resource("prompts/domains/swe/agent_success_criteria.txt").strip()
    )
    with pytest.raises(ValueError, match="unknown-benchmark"):
        prompt_config.render_domain_success_criteria("unknown-benchmark")


def test_gsn_prompt_config_accepts_an_evidence_edge_label_subset() -> None:
    # This verifies graph-construction prompt configs can restrict the evidence relationship vocabulary.
    prompt_config = GSNPromptConfig.model_validate(
        {
            "evidence_edge_labels": ["supports", "undermines"],
            "goal_decompose_instruction": "prompts/not_used.j2",
            "gather_evidence_instruction": "prompts/not_used.j2",
            "assign_confidence_to_goal_instruction": "prompts/not_used.j2",
            "assign_confidence_to_evidence_instruction": "prompts/not_used.j2",
        }
    )

    assert prompt_config.evidence_edge_labels == ["supports", "undermines"]


def test_domain_system_prompt_includes_configured_success_criteria() -> None:
    # This verifies the graph construction prompt embeds configured criteria from its configuration.
    prompt_config = GSNPromptConfig.model_validate(
        {
            "goal_decompose_instruction": "prompts/not_used.j2",
            "gather_evidence_instruction": "prompts/not_used.j2",
            "assign_confidence_to_goal_instruction": "prompts/not_used.j2",
            "assign_confidence_to_evidence_instruction": "prompts/not_used.j2",
        }
    )
    rendered = resolve_template("prompts/gsn/direct/agentic/system_prompt.j2").render(
        overall_goal="Successful Final Outcome",
        overall_goal_auditable_claim="The task succeeded.",
        edge_relationship_types={},
        domain_success_criteria=prompt_config.render_domain_success_criteria(),
    )

    assert "## The Agent's Success Criteria" in rendered
    assert "It has introduced no regressions into the repository." in rendered


def test_supported_confidence_estimator_config_rejects_missing_estimator_type() -> None:
    # This verifies full batch-run configs must opt into an estimator variant explicitly.
    with pytest.raises(ValidationError):
        TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python({"openhands": {}})


def test_supported_confidence_estimator_config_rejects_unknown_estimator_type() -> None:
    # This verifies unsupported estimator variants fail during config validation rather than at runtime.
    with pytest.raises(ValidationError):
        TypeAdapter(SupportedConfidenceEstimatorConfig).validate_python({"estimator_type": "unknown"})
