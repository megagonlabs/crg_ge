import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from litellm.types.utils import TopLogprob

from crg_ce.estimators.base_estimator import ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.openhands.config import HuggingFaceDatasetConfig, LiteLLMVerbalEstimatorConfig
from crg_ce.estimators.openhands.litellm_verbal_estimator import (
    LiteLLMVerbalEstimator,
    StructuredConfidence,
    confidence_from_top_logprobs,
    confidence_from_true_or_false_samples,
    parse_confidence_percentage,
)
from crg_ce.resources import read_resource
from crg_ce.utils.openhands import load_conversation_state_and_events_from_archive
from crg_ce.utils.openhands_trajectory import render_trajectory

ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/agronholm__typeguard.b6a7e438.combine_module__tr31kstl.tar.gz")
PROMPT_FIXTURES_DIR = Path("src/crg_ce/estimators/openhands/test_data/litellm_verbal_prompts")
V3_DATASET_CONFIG = HuggingFaceDatasetConfig(path="Brendan/openhands_ce_data_v3", split="valid")
DOMAIN_CRITERIA_PATHS = {
    "swe-smith": "prompts/domains/swe/agent_success_criteria.txt",
    "enterprise-ops-gym": "prompts/domains/enterprise/agent_success_criteria.txt",
}
TEST_DOMAIN_SUCCESS_CRITERIA = {"by_benchmark": {"test-benchmark": "prompts/domains/swe/agent_success_criteria.txt"}}


def _input(output_dir: Path) -> ConfEstimationInput:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    return ConfEstimationInput(
        conversation_archive_path=ARCHIVE_PATH,
        output_dir=output_dir,
        instance_id="test-instance",
        model="trajectory-model",
        problem_statement="test-problem-statement",
        benchmark="test-benchmark",
    )


def test_parse_confidence_percentage_requires_valid_final_line() -> None:
    # This verifies the pure parser accepts whole-number percentages and rejects decimals and out-of-range values.
    assert parse_confidence_percentage("Reasoning\nConfidence: 82%") == 0.82

    with pytest.raises(ValueError, match="whole-number"):
        parse_confidence_percentage("Confidence: 0.9%")

    with pytest.raises(ValueError, match="between 0 and 100"):
        parse_confidence_percentage("Confidence: 120%")


def test_litellm_verbal_config_requires_matching_benchmark_success_criteria() -> None:
    # This verifies LiteLLM verbal estimation has no implicit criteria or benchmark fallback.
    cfg = LiteLLMVerbalEstimatorConfig.model_validate({"domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA})

    with pytest.raises(ValueError, match="requires a benchmark"):
        cfg.render_domain_success_criteria(None)
    with pytest.raises(ValueError, match="unknown-benchmark"):
        cfg.render_domain_success_criteria("unknown-benchmark")


def test_confidence_from_top_logprobs_strips_tokens_and_normalizes_true_probability() -> None:
    # This verifies whitespace variants are marginalized by label before True/False probability normalization.
    confidence = confidence_from_top_logprobs(
        [
            TopLogprob(token="True", logprob=math.log(0.4)),
            TopLogprob(token=" True", logprob=math.log(0.2)),
            TopLogprob(token=" False", logprob=math.log(0.2)),
            TopLogprob(token="False", logprob=math.log(0.1)),
            TopLogprob(token="Maybe", logprob=math.log(0.1)),
        ]
    )

    assert confidence == pytest.approx(2 / 3)


def test_confidence_from_true_or_false_samples_requires_exact_binary_samples() -> None:
    # This verifies sampled binary confidence permits reasoning and excludes up to two malformed answers.
    assert (
        confidence_from_true_or_false_samples(
            ["Reasoning\nSuccessful: True"] * 7 + ["Reasoning\nSuccessful: False"] + ["Maybe"] * 2, 10
        )
        == 0.875
    )

    with pytest.raises(ValueError, match="Expected 10"):
        confidence_from_true_or_false_samples(["Successful: True"] * 9, 10)
    with pytest.raises(ValueError, match="3 malformed"):
        confidence_from_true_or_false_samples(["Successful: True"] * 7 + ["Successful: Maybe"] * 3, 10)


def test_structured_estimation_renders_trajectory_and_normalizes_score(monkeypatch, tmp_path: Path) -> None:
    # This verifies structured mode evaluates the rendered trajectory with the configured evaluator and scale.
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    calls: list[dict] = []

    def fake_complete_structured(**kwargs):
        calls.append(kwargs)
        kwargs["stats"].record_usage(
            model=kwargs["model"],
            calls=1,
            prompt_tokens=10,
            completion_tokens=2,
            reasoning_tokens=0,
            total_tokens=12,
            cost=0.01,
        )
        return StructuredConfidence(confidence=3, rationale="The patch appears correct.")

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.complete_structured",
        fake_complete_structured,
    )
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator"},
            "verbalization": {"scale_min": 1, "scale_max": 5},
            "domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA,
        }
    )

    output = LiteLLMVerbalEstimator(cfg).estimate_confidence(_input(tmp_path))

    assert output.confidence == 0.5
    assert output.total_tokens == 12
    assert output.generated_tokens == 2
    assert output.cost == 0.01
    assert output.usage_by_model["openai/evaluator"].calls == 1
    assert output.usage_by_model["openai/evaluator"].total_tokens == 12
    assert calls[0]["model"] == "openai/evaluator"
    assert "rendered trajectory" in calls[0]["messages"][0]["content"]
    assert ConfEstimationOutput.model_validate_json((tmp_path / "output.json").read_text()) == output


def test_ask_and_parse_retries_invalid_responses_and_accumulates_usage(monkeypatch, tmp_path: Path) -> None:
    # This verifies parse mode retries malformed percentages, forwards the configured timeout, reports all usage,
    # and logs the unchanged prompt once.
    monkeypatch.setenv("LOG_PROMPTS", "true")
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    responses = iter(["Confidence: high", "Reasoning\nConfidence: 82%"])
    calls: list[dict] = []

    def fake_complete_text(**kwargs):
        calls.append(kwargs)
        kwargs["stats"].record_usage(
            model=kwargs["model"],
            calls=1,
            prompt_tokens=7,
            completion_tokens=3,
            reasoning_tokens=0,
            total_tokens=10,
            cost=0.02,
        )
        return next(responses)

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.complete_text",
        fake_complete_text,
    )
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator", "timeout": 1200},
            "query_type": "ask_and_parse",
            "domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA,
        }
    )

    output = LiteLLMVerbalEstimator(cfg).estimate_confidence(_input(tmp_path))

    assert output.confidence == 0.82
    assert output.total_tokens == 20
    assert output.generated_tokens == 6
    assert output.cost == 0.04
    assert output.usage_by_model["openai/evaluator"].calls == 2
    assert output.usage_by_model["openai/evaluator"].total_tokens == 20
    assert len(calls) == 2
    assert all(call["timeout"] == 1200 for call in calls)
    assert [path.name for path in (tmp_path / "prompts").iterdir()] == ["0.txt"]


def test_ask_and_parse_includes_benchmark_specific_success_criteria(monkeypatch, tmp_path: Path) -> None:
    # This verifies LiteLLM verbalization evaluates trajectories against the criteria configured for their benchmark.
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    calls: list[dict] = []

    def fake_complete_text(**kwargs):
        calls.append(kwargs)
        return "Confidence: 82%"

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.complete_text",
        fake_complete_text,
    )
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator"},
            "query_type": "ask_and_parse",
            "domain_success_criteria": {
                "by_benchmark": {"test-benchmark": "prompts/domains/swe/agent_success_criteria.txt"}
            },
        }
    )
    ce_input = _input(tmp_path)
    ce_input.benchmark = "test-benchmark"

    LiteLLMVerbalEstimator(cfg).estimate_confidence(ce_input)

    assert "It has introduced no regressions into the repository." in calls[0]["messages"][0]["content"]


@pytest.mark.integration
@pytest.mark.parametrize("benchmark", DOMAIN_CRITERIA_PATHS)
def test_ask_and_parse_first_real_prompt_matches_benchmark_fixture(benchmark: str) -> None:
    # This verifies each benchmark's first v3 trajectory produces a stable, readable evaluator prompt from its first
    # ten events.
    fixture = PROMPT_FIXTURES_DIR / f"{benchmark}.txt"
    if not fixture.is_file():
        pytest.skip("This test requires a locally supplied benchmark prompt fixture.")
    item = next(item for item in V3_DATASET_CONFIG.load_dataset() if item["benchmark"] == benchmark)
    _, events = load_conversation_state_and_events_from_archive(
        V3_DATASET_CONFIG.base_path / item["conversation_archive_path"]
    )
    trajectory = render_trajectory(events, start_at_first_action_event=True)
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "query_type": "ask_and_parse",
            "domain_success_criteria": {"by_benchmark": DOMAIN_CRITERIA_PATHS},
        }
    )
    estimator = LiteLLMVerbalEstimator(cfg)
    prompt = estimator._render_instruction(
        trajectory,
        problem_statement=item["problem_statement"],
        domain_success_criteria=cfg.render_domain_success_criteria(benchmark),
        output_instruction=estimator.ask_and_parse_output_instruction,
    )
    assert prompt == fixture.read_text().removesuffix("\n")


def test_default_ask_and_parse_output_instruction_matches_legacy_fixture() -> None:
    # This verifies the configurable default retains the exact legacy instruction appended to evaluator prompts.
    cfg = LiteLLMVerbalEstimatorConfig()

    assert read_resource(cfg.ask_and_parse_output_instruction).strip() == (
        PROMPT_FIXTURES_DIR / "ask_and_parse_output_instruction.txt"
    ).read_text().removesuffix("\n")


def test_true_or_false_retries_until_both_tokens_are_present(monkeypatch, tmp_path: Path) -> None:
    # This verifies binary mode uses continuation settings and retries until top-ten logprobs contain both labels.
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    top_logprobs_by_attempt = iter(
        [
            [TopLogprob(token=" True", logprob=math.log(0.9))],
            [
                TopLogprob(token=" True", logprob=math.log(0.8)),
                TopLogprob(token=" False", logprob=math.log(0.2)),
            ],
        ]
    )
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    logprobs=SimpleNamespace(content=[SimpleNamespace(top_logprobs=next(top_logprobs_by_attempt))])
                )
            ],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=1, total_tokens=10),
            response_cost=0.01,
        )

    monkeypatch.setattr("crg_ce.estimators.openhands.litellm_verbal_estimator.litellm.completion", fake_completion)
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator"},
            "query_type": "true_or_false",
            "domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA,
        }
    )

    output = LiteLLMVerbalEstimator(cfg).estimate_confidence(_input(tmp_path))

    assert output.confidence == pytest.approx(0.8)
    assert output.total_tokens == 20
    assert output.generated_tokens == 2
    assert output.cost == pytest.approx(0.02)
    assert output.usage_by_model["openai/evaluator"].calls == 2
    assert output.usage_by_model["openai/evaluator"].total_tokens == 20
    assert len(calls) == 2
    assert calls[0]["messages"][-1] == {"role": "assistant", "content": "Answer:"}
    assert "rendered trajectory" in calls[0]["messages"][0]["content"]
    assert calls[0]["temperature"] == 0
    assert calls[0]["max_completion_tokens"] == 1
    assert calls[0]["logprobs"] is True
    assert calls[0]["top_logprobs"] == 10
    assert calls[0]["extra_body"] == {
        "continue_final_message": True,
        "add_generation_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_sampled_true_or_false_counts_ten_litellm_choices(monkeypatch, tmp_path: Path) -> None:
    # This verifies sampled binary mode uses the base verbalizer, requests ten choices, and scores their True fraction.
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        samples = ["Reasoning\nSuccessful: True"] * 7 + ["Reasoning\nSuccessful: False"] * 3
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=sample)) for sample in samples],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=10, total_tokens=19),
            response_cost=0.01,
        )

    monkeypatch.setattr("crg_ce.estimators.openhands.litellm_verbal_estimator.litellm.completion", fake_completion)
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator", "completion_kwargs": {"temperature": 1.0}},
            "query_type": "sampled_true_or_false",
            "true_or_false_samples": 10,
            "domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA,
        }
    )

    output = LiteLLMVerbalEstimator(cfg).estimate_confidence(_input(tmp_path))

    assert output.confidence == 0.7
    assert output.total_tokens == 19
    assert output.generated_tokens == 10
    assert output.usage_by_model["openai/evaluator"].calls == 1
    assert len(calls) == 1
    assert calls[0]["n"] == 10
    assert calls[0]["temperature"] == 1.0
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert "rendered trajectory" in calls[0]["messages"][0]["content"]
    assert '"Successful: <answer>"' in calls[0]["messages"][0]["content"]


def test_sampled_true_or_false_retries_a_malformed_batch(monkeypatch, tmp_path: Path) -> None:
    # This verifies three malformed choices retry the complete ten-choice request under the estimator's retry policy.
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    calls: list[dict] = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        samples = ["Successful: True"] * 6 + ["Successful: False"] * 4
        if len(calls) == 1:
            samples[-3:] = ["I cannot decide"] * 3
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=sample)) for sample in samples],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=10, total_tokens=19),
            response_cost=0.01,
        )

    monkeypatch.setattr("crg_ce.estimators.openhands.litellm_verbal_estimator.litellm.completion", fake_completion)
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator"},
            "query_type": "sampled_true_or_false",
            "true_or_false_samples": 10,
            "domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA,
        }
    )

    output = LiteLLMVerbalEstimator(cfg).estimate_confidence(_input(tmp_path))

    assert output.confidence == 0.6
    assert output.usage_by_model["openai/evaluator"].calls == 2
    assert len(calls) == 2


def test_sampled_true_or_false_fails_after_three_malformed_batches(monkeypatch, tmp_path: Path) -> None:
    # This verifies an instance fails only after all three ten-choice batches contain at least three malformed answers.
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.litellm_verbal_estimator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    calls = 0

    def fake_completion(**kwargs):
        nonlocal calls
        calls += 1
        samples = ["Successful: True"] * 7 + ["I cannot decide"] * 3
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=sample)) for sample in samples],
            usage=SimpleNamespace(prompt_tokens=9, completion_tokens=10, total_tokens=19),
            response_cost=0.01,
        )

    monkeypatch.setattr("crg_ce.estimators.openhands.litellm_verbal_estimator.litellm.completion", fake_completion)
    cfg = LiteLLMVerbalEstimatorConfig.model_validate(
        {
            "agent": {"model_name": "openai/evaluator"},
            "query_type": "sampled_true_or_false",
            "true_or_false_samples": 10,
            "domain_success_criteria": TEST_DOMAIN_SUCCESS_CRITERIA,
        }
    )

    with pytest.raises(ValueError, match="Could not obtain valid confidence after 3 attempts"):
        LiteLLMVerbalEstimator(cfg).estimate_confidence(_input(tmp_path))

    assert calls == 3
