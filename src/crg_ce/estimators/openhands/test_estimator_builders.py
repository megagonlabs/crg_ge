from pathlib import Path

from crg_ce.estimators.calibration import (
    CalibratedConfidenceEstimator,
    CalibrationConfig,
    TemperatureCalibrator,
)
from crg_ce.estimators.openhands.builders import build_openhands_estimator
from crg_ce.estimators.openhands.config import (
    BasicLiteLLMVerbalEstimatorConfig,
    GSNPlainVerbalizedEstimatorConfig,
    GSNPostHocAggregateConfig,
    LiteLLMVerbalEstimatorConfig,
    OHGSNEstimatorConfig,
)
from crg_ce.estimators.openhands.gsn_plain_verbalized_estimator import GSNPlainVerbalizedEstimator
from crg_ce.estimators.openhands.litellm_verbal_estimator import LiteLLMVerbalEstimator
from crg_ce.estimators.openhands.oh_gsn_estimator import (
    GSNPostHocAggregateEstimator,
    OHGSNConfidenceEstimator,
)
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import (
    GSNGraphGeneratorConfig,
    GSNGraphPopulatorConfig,
)


def _gsn_cfg() -> OHGSNEstimatorConfig:
    prompts = {
        "gather_evidence_method": "all_at_once",
        "goal_decompose_instruction": "prompts/gsn/direct/agentic/goal_decomposition.j2",
        "gather_evidence_instruction": "prompts/gsn/direct/agentic/evidence_gathering.j2",
        "assign_confidence_to_goal_instruction": (
            "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
        ),
        "assign_confidence_to_evidence_instruction": (
            "prompts/gsn/direct/agentic/estimation/assign_confidence_to_evidence.j2"
        ),
    }
    return OHGSNEstimatorConfig(
        graph_generator=GSNGraphGeneratorConfig.model_validate(
            {
                "generator_type": "agentic",
                "agent": {
                    "model_name": "openai/test-model",
                    "tools_preset": "gsn_agentic_graph_construction",
                },
                "prompts": prompts,
            }
        ),
        graph_populator=GSNGraphPopulatorConfig.model_validate(
            {
                "generator_type": "litellm",
                "agent": {"model_name": "openai/test-model", "tools_preset": "default"},
                "prompts": prompts,
            }
        ),
    )


def test_build_openhands_estimator_returns_litellm_verbal_estimator() -> None:
    # This verifies the factory routes the ex-situ tagged config to the direct LiteLLM estimator.
    estimator = build_openhands_estimator(
        BasicLiteLLMVerbalEstimatorConfig(
            litellm=LiteLLMVerbalEstimatorConfig(
                domain_success_criteria={"by_benchmark": {"test": "prompts/domains/swe/agent_success_criteria.txt"}}  # type: ignore
            )
        )
    )

    assert isinstance(estimator, LiteLLMVerbalEstimator)


def test_build_openhands_estimator_returns_gsn_estimator() -> None:
    # This verifies the factory routes GSN tagged configs to the graph-backed confidence estimator.
    estimator = build_openhands_estimator(_gsn_cfg())

    assert isinstance(estimator, OHGSNConfidenceEstimator)


def test_build_openhands_estimator_returns_gsn_post_hoc_aggregate_estimator() -> None:
    # This verifies aggregate-tagged configs route to deterministic post-hoc graph aggregation.
    estimator = build_openhands_estimator(
        GSNPostHocAggregateConfig(replay_from=Path("runs/source-gsn.yaml"), aggregation_type="product")
    )

    assert isinstance(estimator, GSNPostHocAggregateEstimator)


def test_build_openhands_estimator_returns_gsn_plain_verbalized_estimator() -> None:
    # This verifies the factory routes loaded-graph verbalization configs to the single-call GSN baseline.
    estimator = build_openhands_estimator(GSNPlainVerbalizedEstimatorConfig(load_graphs_from_path=Path("graphs")))

    assert isinstance(estimator, GSNPlainVerbalizedEstimator)


def test_build_openhands_estimator_wraps_configured_calibration() -> None:
    # This verifies the shared estimator factory applies calibration generically rather than within each estimator.
    config = BasicLiteLLMVerbalEstimatorConfig(
        litellm=LiteLLMVerbalEstimatorConfig(),
        calibration=CalibrationConfig(mode="temperature", temperature=2.0),
    )

    estimator = build_openhands_estimator(config)

    assert isinstance(estimator, CalibratedConfidenceEstimator)
    assert isinstance(estimator.estimator, LiteLLMVerbalEstimator)
    assert isinstance(estimator.calibrator, TemperatureCalibrator)
