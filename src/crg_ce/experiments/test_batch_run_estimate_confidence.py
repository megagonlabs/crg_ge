import asyncio
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from crg_ce.estimators.base_estimator import BaseConfidenceEstimator, ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.openhands.config import (
    AgentConfig,
    BasicLiteLLMVerbalEstimatorConfig,
    GSNPostHocAggregateConfig,
    HuggingFaceDatasetConfig,
    LiteLLMVerbalEstimatorConfig,
    OutputConfig,
)
from crg_ce.experiments.batch_run_estimate_confidence import (
    ConfidenceEstimationRunConfig,
    _allow_qwen_3_8_reasoning_effort,
    estimate_dataset_confidence,
    estimate_dataset_confidence_async,
    estimate_dataset_confidence_async_batched,
    parse_args,
)


class FakeEstimator(BaseConfidenceEstimator):
    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        return ConfEstimationOutput(confidence=0.42, total_tokens=100, generated_tokens=20, cost=0.01)


def _no_agent_estimator_config() -> GSNPostHocAggregateConfig:
    return GSNPostHocAggregateConfig(replay_from=Path("runs/source-gsn.yaml"), aggregation_type="product")


def _agent_estimator_config(agent: AgentConfig | None = None) -> BasicLiteLLMVerbalEstimatorConfig:
    return BasicLiteLLMVerbalEstimatorConfig(litellm=LiteLLMVerbalEstimatorConfig(agent=agent or AgentConfig()))


def test_confidence_estimation_output_defaults_usage_to_unavailable() -> None:
    # This verifies estimators that have not implemented usage reporting remain valid with explicit sentinel values.
    output = ConfEstimationOutput(confidence=0.42)

    assert output.total_tokens == -1
    assert output.generated_tokens == -1
    assert output.cost == -1


def test_batch_run_allows_reasoning_effort_for_qwen_3_8(caplog, tmp_path: Path) -> None:
    # This verifies batch_run_ce adds the LiteLLM compatibility allowlist for Qwen 3.8 exactly once, without
    # requiring every run config to repeat it; it assumes agent configs are mutable Pydantic models.
    agent = AgentConfig(model_name="openai/Qwen/Qwen3.8-27B-FP8")
    cfg = ConfidenceEstimationRunConfig(
        estimator=_agent_estimator_config(agent),
        output=OutputConfig(output_dir=tmp_path),
    )

    _allow_qwen_3_8_reasoning_effort(cfg)
    _allow_qwen_3_8_reasoning_effort(cfg)

    assert agent.allowed_openai_params == ["reasoning_effort"]
    assert caplog.messages == [
        "Added allowed_openai_params=['reasoning_effort'] for Qwen 3.8 model openai/Qwen/Qwen3.8-27B-FP8"
    ]


def test_base_async_batch_delegates_to_each_existing_async_estimate(tmp_path: Path) -> None:
    # This verifies estimators inherit a useful batch implementation without changing their single-item async logic.
    seen_instance_ids: list[str] = []

    class AsyncFakeEstimator(BaseConfidenceEstimator):
        def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
            raise AssertionError("The synchronous estimator path must not be used")

        async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
            seen_instance_ids.append(ce_input.instance_id)
            return ConfEstimationOutput(confidence=0.5)

    archive_path = tmp_path / "conversation.tar.gz"
    archive_path.write_bytes(b"archive")
    ce_inputs = [
        ConfEstimationInput(
            conversation_archive_path=archive_path,
            output_dir=tmp_path / f"output-{index}",
            instance_id=f"instance-{index}",
            model="test-model",
            problem_statement="Fix the reported bug.",
        )
        for index in range(2)
    ]

    outputs = asyncio.run(AsyncFakeEstimator().aestimate_confidence_batch(ce_inputs))

    assert seen_instance_ids == ["instance-0", "instance-1"]
    assert outputs == [ConfEstimationOutput(confidence=0.5), ConfEstimationOutput(confidence=0.5)]


def test_estimate_dataset_confidence_uses_estimator_factory(monkeypatch, tmp_path: Path) -> None:
    # This verifies batch estimation depends only on the shared estimator interface and passes the dataset-declared
    # trajectory type through to the estimator, rather than inferring it from the path.
    # Create a fake conversation.tar.gz that resolves under tmp_path through HuggingFaceDatasetConfig.base_path.
    archive_path = tmp_path / "repo" / "train" / "conversation.tar.gz"
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"archive")
    dataset_cfg = HuggingFaceDatasetConfig(path="repo", split="train")
    estimator_cfg = _agent_estimator_config()
    calls = []
    received_inputs: list[ConfEstimationInput] = []

    monkeypatch.setenv("DATA_BASE_PATH", str(tmp_path))
    # Mock a 1-item Hugging Face dataset whose relative archive path matches the fake archive above.
    monkeypatch.setattr(
        HuggingFaceDatasetConfig,
        "load_dataset",
        lambda self: SimpleNamespace(
            to_list=lambda: [
                {
                    "conversation_archive_path": "conversation.tar.gz",
                    "instance_id": "test-instance",
                    "model": "test-model",
                    "problem_statement": "Fix the reported bug.",
                    "benchmark": "test-benchmark",
                    "trajectory_type": "acp",
                }
            ]
        ),
    )

    def fake_build_openhands_estimator(cfg):
        calls.append(cfg)

        class CapturingFakeEstimator(FakeEstimator):
            def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
                received_inputs.append(ce_input)
                return super().estimate_confidence(ce_input)

        return CapturingFakeEstimator()

    monkeypatch.setattr(
        "crg_ce.experiments.batch_run_estimate_confidence.build_openhands_estimator",
        fake_build_openhands_estimator,
    )

    results = estimate_dataset_confidence(
        estimator_cfg=estimator_cfg,
        dataset_cfg=dataset_cfg,
        output_dir=tmp_path,
        max_workers=1,
    )

    assert calls == [estimator_cfg]
    assert received_inputs[0].problem_statement == "Fix the reported bug."
    assert received_inputs[0].benchmark == "test-benchmark"
    assert received_inputs[0].trajectory_type == "acp"
    assert results == [
        {
            "conversation_archive_path": "conversation.tar.gz",
            "instance_id": "test-instance",
            "model": "test-model",
            "problem_statement": "Fix the reported bug.",
            "benchmark": "test-benchmark",
            "trajectory_type": "acp",
            "estimated_confidence": 0.42,
            "total_tokens": 100,
            "generated_tokens": 20,
            "cost": 0.01,
        }
    ]


def test_parse_args_accepts_resume_flag() -> None:
    # This verifies the CLI exposes resume while retaining the required run-config path.
    args = parse_args(["runs/test.yaml", "--resume"])

    assert args.config == Path("runs/test.yaml")
    assert args.resume is True


def test_async_execution_configuration_is_explicit_and_coherent() -> None:
    # This verifies asynchronous runs require a limit on every configured agent, while no-agent estimators need none.
    base = {"estimator": _no_agent_estimator_config()}
    agent_estimator = _agent_estimator_config()
    limited_agent_estimator = _agent_estimator_config(AgentConfig(max_concurrent_llm_calls=2))

    assert ConfidenceEstimationRunConfig.model_validate(base).execution_mode == "sync"
    assert ConfidenceEstimationRunConfig.model_validate({**base, "execution_mode": "async"}).execution_mode == "async"
    with pytest.raises(ValidationError, match="required for every agent"):
        ConfidenceEstimationRunConfig.model_validate({"estimator": agent_estimator, "execution_mode": "async"})
    assert (
        ConfidenceEstimationRunConfig.model_validate(
            {"estimator": limited_agent_estimator, "execution_mode": "async_batched"}
        ).execution_mode
        == "async_batched"
    )
    with pytest.raises(ValidationError, match="must not be set"):
        ConfidenceEstimationRunConfig.model_validate({"estimator": limited_agent_estimator})


def test_async_batched_execution_uses_max_workers_as_fixed_batch_size(monkeypatch, tmp_path: Path) -> None:
    # This verifies batch mode creates ordered cohorts of exactly max_workers inputs and shares one limiter across
    # cohorts. It assumes the fake estimator represents a native batch implementation with no archive parsing.
    items = []
    for index in range(5):
        archive_name = f"conversation-{index}.tar.gz"
        archive_path = tmp_path / "repo" / "train" / archive_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        items.append(
            {
                "conversation_archive_path": archive_name,
                "instance_id": f"instance-{index}",
                "model": "test-model",
                "problem_statement": "Fix the reported bug.",
                "benchmark": "test-benchmark",
            }
        )
    dataset_cfg = HuggingFaceDatasetConfig(path="repo", split="train")
    estimator_cfg = _agent_estimator_config(AgentConfig(max_concurrent_llm_calls=1))
    monkeypatch.setenv("DATA_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(HuggingFaceDatasetConfig, "load_dataset", lambda self: SimpleNamespace(to_list=lambda: items))
    batch_sizes: list[int] = []
    received_limiters = []

    class BatchFakeEstimator(BaseConfidenceEstimator):
        def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
            raise AssertionError("The batch implementation must be used")

        async def aestimate_confidence_batch(
            self, ce_inputs: Sequence[ConfEstimationInput]
        ) -> list[ConfEstimationOutput]:
            batch_sizes.append(len(ce_inputs))
            return [ConfEstimationOutput(confidence=0.5) for _ in ce_inputs]

    def fake_builder(cfg, *, llm_limiters):
        received_limiters.append(llm_limiters)
        return BatchFakeEstimator()

    monkeypatch.setattr(
        "crg_ce.experiments.batch_run_estimate_confidence.build_openhands_estimator",
        fake_builder,
    )
    results = asyncio.run(
        estimate_dataset_confidence_async_batched(
            estimator_cfg=estimator_cfg,
            dataset_cfg=dataset_cfg,
            output_dir=tmp_path,
            max_workers=2,
        )
    )

    assert batch_sizes == [2, 2, 1]
    assert len({id(limiters) for limiters in received_limiters}) == 1
    assert len(received_limiters[0]) == 1
    assert [result["instance_id"] for result in results] == [item["instance_id"] for item in items]


def test_async_batched_execution_keeps_successes_when_one_item_times_out(
    monkeypatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # This verifies a timeout returned for one batch item is recorded without discarding a successful peer. It
    # assumes the estimator preserves input order when returning outputs and item-local exceptions.
    items = [
        {
            "conversation_archive_path": f"conversation-{index}.tar.gz",
            "instance_id": f"instance-{index}",
            "model": "test-model",
            "problem_statement": "Fix the reported bug.",
            "benchmark": "test-benchmark",
        }
        for index in range(2)
    ]
    for item in items:
        archive_path = tmp_path / "repo" / "train" / item["conversation_archive_path"]
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
    dataset_cfg = HuggingFaceDatasetConfig(path="repo", split="train")
    estimator_cfg = _agent_estimator_config(AgentConfig(max_concurrent_llm_calls=1))
    monkeypatch.setenv("DATA_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(HuggingFaceDatasetConfig, "load_dataset", lambda self: SimpleNamespace(to_list=lambda: items))

    class PartiallyFailingEstimator(FakeEstimator):
        async def aestimate_confidence_batch(self, ce_inputs):
            try:
                raise TimeoutError("request timed out")
            except TimeoutError as error:
                return [ConfEstimationOutput(confidence=0.75), error]

    monkeypatch.setattr(
        "crg_ce.experiments.batch_run_estimate_confidence.build_openhands_estimator",
        lambda cfg, *, llm_limiters: PartiallyFailingEstimator(),
    )
    caplog.set_level(logging.ERROR)

    results = asyncio.run(
        estimate_dataset_confidence_async_batched(
            estimator_cfg=estimator_cfg,
            dataset_cfg=dataset_cfg,
            output_dir=tmp_path,
            max_workers=2,
        )
    )

    assert [result["instance_id"] for result in results] == ["instance-0"]
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert failures[0]["instance_id"] == "instance-1"
    assert failures[0]["model"] == "test-model"
    assert failures[0]["problem_statement"] == "Fix the reported bug."
    assert failures[0]["error"] == "TimeoutError: request timed out"
    assert "raise TimeoutError" in failures[0]["traceback"]
    assert failures[0]["conversation_archive_path"] == str((tmp_path / "repo/train/conversation-1.tar.gz").resolve())
    assert failures[0]["log_prob_features_path"] == str(
        (tmp_path / "instance-1/test-model/log_prob_features.json").resolve()
    )
    assert failures[0]["log_prob_prompt_path"] == str(
        (tmp_path / "instance-1/test-model/log_prob_prompt.json").resolve()
    )
    assert failures[0]["dataset_item"] == items[1]
    assert "Problem statement:\nFix the reported bug." in caplog.text


def test_async_batch_bounds_problem_workflows_and_llm_calls(monkeypatch, tmp_path: Path) -> None:
    # This verifies the outer limit covers whole problem lifecycles while the independent inner limit covers only
    # simulated LLM calls. It assumes each fake item has a valid archive path but performs no archive parsing.
    item_count = 6
    items = []
    for index in range(item_count):
        archive_name = f"conversation-{index}.tar.gz"
        archive_path = tmp_path / "repo" / "train" / archive_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(b"archive")
        items.append(
            {
                "conversation_archive_path": archive_name,
                "instance_id": f"instance-{index}",
                "model": "test-model",
                "problem_statement": "Fix the reported bug.",
                "benchmark": "test-benchmark",
            }
        )

    dataset_cfg = HuggingFaceDatasetConfig(path="repo", split="train")
    estimator_cfg = _agent_estimator_config(AgentConfig(max_concurrent_llm_calls=1))
    monkeypatch.setenv("DATA_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        HuggingFaceDatasetConfig,
        "load_dataset",
        lambda self: SimpleNamespace(to_list=lambda: items),
    )
    active_problems = 0
    max_active_problems = 0
    active_llm_calls = 0
    max_active_llm_calls = 0

    def fake_builder(cfg, *, llm_limiters):
        llm_limiter = next(iter(llm_limiters.values()))

        class AsyncFakeEstimator(BaseConfidenceEstimator):
            def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
                raise AssertionError("The synchronous estimator path must not be used")

            async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
                nonlocal active_problems, max_active_problems, active_llm_calls, max_active_llm_calls
                active_problems += 1
                max_active_problems = max(max_active_problems, active_problems)
                try:
                    await asyncio.sleep(0.01)
                    async with llm_limiter.slot():
                        active_llm_calls += 1
                        max_active_llm_calls = max(max_active_llm_calls, active_llm_calls)
                        try:
                            await asyncio.sleep(0.01)
                        finally:
                            active_llm_calls -= 1
                    await asyncio.sleep(0.01)
                    return ConfEstimationOutput(confidence=0.5)
                finally:
                    active_problems -= 1

        return AsyncFakeEstimator()

    monkeypatch.setattr(
        "crg_ce.experiments.batch_run_estimate_confidence.build_openhands_estimator",
        fake_builder,
    )
    results = asyncio.run(
        estimate_dataset_confidence_async(
            estimator_cfg=estimator_cfg,
            dataset_cfg=dataset_cfg,
            output_dir=tmp_path,
            max_workers=2,
        )
    )

    assert len(results) == item_count
    assert max_active_problems == 2
    assert max_active_llm_calls == 1


def test_resume_uses_estimator_saved_output(monkeypatch, tmp_path: Path, caplog) -> None:
    # This verifies the outer loop delegates saved-output lookup to the estimator and skips only when it returns one.
    dataset_cfg = HuggingFaceDatasetConfig(path="repo", split="train")
    estimator_cfg = _agent_estimator_config()
    item = {
        "conversation_archive_path": "conversation.tar.gz",
        "instance_id": "test-instance",
        "model": "test-model",
        "problem_statement": "Fix the reported bug.",
        "benchmark": "test-benchmark",
    }
    archive_path = tmp_path / "repo" / "train" / item["conversation_archive_path"]
    archive_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"archive")
    monkeypatch.setenv("DATA_BASE_PATH", str(tmp_path))
    monkeypatch.setattr(
        HuggingFaceDatasetConfig,
        "load_dataset",
        lambda self: SimpleNamespace(to_list=lambda: [item]),
    )
    monkeypatch.setattr(
        "crg_ce.experiments.batch_run_estimate_confidence.build_openhands_estimator",
        lambda cfg: FakeEstimator(),
    )
    output_path = tmp_path / item["instance_id"] / item["model"] / "output.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text(ConfEstimationOutput(confidence=0.73).model_dump_json(indent=2))
    with caplog.at_level(logging.INFO):
        results = estimate_dataset_confidence(
            estimator_cfg=estimator_cfg,
            dataset_cfg=dataset_cfg,
            output_dir=tmp_path,
            max_workers=1,
            resume=True,
        )

    assert results == [
        {
            **item,
            "estimated_confidence": 0.73,
            "total_tokens": -1,
            "generated_tokens": -1,
            "cost": -1,
        }
    ]
    expected_log = "Skipped instance_id=test-instance model=test-model because a saved estimator output was found"
    assert expected_log in caplog.text
