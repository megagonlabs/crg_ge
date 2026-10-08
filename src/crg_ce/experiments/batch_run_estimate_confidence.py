import argparse
import asyncio
import json
import logging
import threading
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial
from pathlib import Path
from typing import Any, Literal, Self, cast

import litellm
import pandas as pd
from datasets import Dataset
from pydantic import Field, model_validator

from crg_ce.datasets.oh_benchmarks.data_types import ConfEstimatedOpenHandsCEDataPoint, OpenHandsCEDataPoint
from crg_ce.estimators.base_estimator import BaseConfidenceEstimator, ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.openhands.builders import build_openhands_estimator
from crg_ce.estimators.openhands.config import (
    BaseRunConfig,
    HuggingFaceDatasetConfig,
    SupportedConfidenceEstimatorConfig,
    agent_configs_for_estimator,
    load_run_config,
)
from crg_ce.llm_concurrency import build_llm_limiters
from crg_ce.metrics import ConfMetrics, metrics_from_df

logger = logging.getLogger(__name__)


class ConfidenceEstimationRunConfig(BaseRunConfig):
    estimator: SupportedConfidenceEstimatorConfig
    dataset: HuggingFaceDatasetConfig = Field(default_factory=HuggingFaceDatasetConfig)
    output_filename: str = Field(default="confidence_estimates.jsonl")
    confidence_column: str = Field(default="estimated_confidence")
    execution_mode: Literal["sync", "async", "async_batched"] = "sync"

    @model_validator(mode="after")
    def validate_execution_settings(self) -> Self:
        agent_configs = agent_configs_for_estimator(self.estimator)
        if self.execution_mode in {"async", "async_batched"}:
            if any(agent.max_concurrent_llm_calls is None for agent in agent_configs):
                raise ValueError("max_concurrent_llm_calls is required for every agent in asynchronous runs")
        elif any(agent.max_concurrent_llm_calls is not None for agent in agent_configs):
            raise ValueError("max_concurrent_llm_calls must not be set when execution_mode is 'sync'")
        return self


def _allow_qwen_3_8_reasoning_effort(cfg: ConfidenceEstimationRunConfig) -> None:
    """Allow LiteLLM to forward reasoning effort for Qwen 3.8 endpoints with incomplete metadata."""
    for agent_cfg in agent_configs_for_estimator(cfg.estimator):
        if "Qwen3.8" not in agent_cfg.model_name or "reasoning_effort" in agent_cfg.allowed_openai_params:
            continue
        agent_cfg.allowed_openai_params.append("reasoning_effort")
        logger.warning(
            "Added allowed_openai_params=['reasoning_effort'] for Qwen 3.8 model %s",
            agent_cfg.model_name,
        )


def _estimate_confidence(
    estimator: BaseConfidenceEstimator,
    ce_input: ConfEstimationInput,
) -> ConfEstimationOutput:
    return estimator.estimate_confidence(ce_input)


async def _aestimate_confidence(
    estimator: BaseConfidenceEstimator,
    ce_input: ConfEstimationInput,
) -> ConfEstimationOutput:
    return await estimator.aestimate_confidence(ce_input)


async def _aestimate_confidence_batch(
    estimator: BaseConfidenceEstimator,
    ce_inputs: list[ConfEstimationInput],
) -> Sequence[ConfEstimationOutput | BaseException]:
    return await estimator.aestimate_confidence_batch(ce_inputs)


def _failure_record(
    *,
    item: OpenHandsCEDataPoint,
    error: BaseException,
    estimator_cfg: SupportedConfidenceEstimatorConfig,
    dataset_cfg: HuggingFaceDatasetConfig,
    output_dir: Path,
) -> dict[str, Any]:
    item_output_dir = output_dir / item["instance_id"] / item["model"]
    conversation_archive_path = Path(dataset_cfg.base_path / item["conversation_archive_path"])
    return {
        "instance_id": item["instance_id"],
        "model": item["model"],
        "benchmark": item.get("benchmark"),
        "problem_statement": item["problem_statement"],
        "error": f"{type(error).__name__}: {error}",
        "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "conversation_archive_path": str(conversation_archive_path.resolve()),
        "item_output_dir": str(item_output_dir.resolve()),
        "log_prob_features_path": str((item_output_dir / "log_prob_features.json").resolve()),
        "log_prob_prompt_path": str((item_output_dir / "log_prob_prompt.json").resolve()),
        "run_config_path": str((output_dir / "run_config.yaml").resolve()),
        "dataset_item": dict(item),
    }


def _write_failures(output_dir: Path, failures: list[dict[str, Any]]) -> None:
    (output_dir / "failures.json").write_text(json.dumps(failures, indent=2))


def estimate_dataset_confidence(
    *,
    estimator_cfg: SupportedConfidenceEstimatorConfig,
    dataset_cfg: HuggingFaceDatasetConfig,
    output_dir: Path,
    max_workers: int,
    resume: bool = False,
) -> list[ConfEstimatedOpenHandsCEDataPoint]:
    dataset: Dataset = dataset_cfg.load_dataset()
    dataset_items = cast(list[OpenHandsCEDataPoint], dataset.to_list())
    thread_local = threading.local()
    results_lock = threading.Lock()
    failures: list[dict[str, Any]] = []
    counts = {"skipped": 0, "completed": 0, "failed": 0}

    def get_thread_estimator() -> BaseConfidenceEstimator:
        estimator = getattr(thread_local, "estimator", None)
        if estimator is None:
            estimator = build_openhands_estimator(estimator_cfg)
            thread_local.estimator = estimator
        return estimator

    def estimate_confidence_for_item(item: OpenHandsCEDataPoint) -> ConfEstimatedOpenHandsCEDataPoint | None:
        # logging.warning(dataset_cfg.base_path / item["conversation_archive_path"])
        try:
            item_output_dir = output_dir / item["instance_id"] / item["model"]
            ce_input: ConfEstimationInput = ConfEstimationInput(
                conversation_archive_path=Path(dataset_cfg.base_path / item["conversation_archive_path"]),
                output_dir=item_output_dir,
                instance_id=item["instance_id"],
                model=item["model"],
                problem_statement=item["problem_statement"],
                benchmark=item.get("benchmark"),
                trajectory_type=item.get("trajectory_type"),
            )
            estimator = get_thread_estimator()
            ce_output = estimator.get_saved_output(item_output_dir) if resume else None
            if ce_output is not None:
                logger.info(
                    "Skipped instance_id=%s model=%s because a saved estimator output was found",
                    item["instance_id"],
                    item["model"],
                )
                status = "skipped"
            else:
                ce_output = _estimate_confidence(estimator, ce_input)
                status = "completed"
        except Exception as error:
            logger.exception(
                "Failed instance_id=%s model=%s\nProblem statement:\n%s",
                item["instance_id"],
                item["model"],
                item["problem_statement"],
            )
            with results_lock:
                counts["failed"] += 1
                failures.append(
                    _failure_record(
                        item=item,
                        error=error,
                        estimator_cfg=estimator_cfg,
                        dataset_cfg=dataset_cfg,
                        output_dir=output_dir,
                    )
                )
                _write_failures(output_dir, failures)
            return None

        with results_lock:
            if status == "skipped":
                counts["skipped"] += 1
            else:
                counts["completed"] += 1
        return cast(
            ConfEstimatedOpenHandsCEDataPoint,
            {
                **item,
                "estimated_confidence": ce_output.confidence,
                "total_tokens": ce_output.total_tokens,
                "generated_tokens": ce_output.generated_tokens,
                "cost": ce_output.cost,
            },
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(copy_context().run, partial(estimate_confidence_for_item, item)) for item in dataset_items
        ]
        results = [future.result() for future in futures]

    _write_failures(output_dir, failures)
    logger.info(
        "Confidence-estimation report: skipped=%d completed=%d failed=%d",
        counts["skipped"],
        counts["completed"],
        counts["failed"],
    )
    return [result for result in results if result is not None]


async def estimate_dataset_confidence_async(
    *,
    estimator_cfg: SupportedConfidenceEstimatorConfig,
    dataset_cfg: HuggingFaceDatasetConfig,
    output_dir: Path,
    max_workers: int,
    resume: bool = False,
) -> list[ConfEstimatedOpenHandsCEDataPoint]:
    dataset: Dataset = dataset_cfg.load_dataset()
    dataset_items = cast(list[OpenHandsCEDataPoint], dataset.to_list())
    problem_semaphore = asyncio.Semaphore(max_workers)
    llm_limiters = build_llm_limiters(agent_configs_for_estimator(estimator_cfg))
    failures: list[dict[str, Any]] = []
    counts = {"skipped": 0, "completed": 0, "failed": 0}

    async def estimate_confidence_for_item(
        item: OpenHandsCEDataPoint,
    ) -> ConfEstimatedOpenHandsCEDataPoint | None:
        async with problem_semaphore:
            try:
                item_output_dir = output_dir / item["instance_id"] / item["model"]
                ce_input = ConfEstimationInput(
                    conversation_archive_path=Path(dataset_cfg.base_path / item["conversation_archive_path"]),
                    output_dir=item_output_dir,
                    instance_id=item["instance_id"],
                    model=item["model"],
                    problem_statement=item["problem_statement"],
                    benchmark=item.get("benchmark"),
                    trajectory_type=item.get("trajectory_type"),
                )
                estimator = build_openhands_estimator(estimator_cfg, llm_limiters=llm_limiters)
                ce_output = estimator.get_saved_output(item_output_dir) if resume else None
                if ce_output is not None:
                    logger.info(
                        "Skipped instance_id=%s model=%s because a saved estimator output was found",
                        item["instance_id"],
                        item["model"],
                    )
                    status = "skipped"
                else:
                    ce_output = await _aestimate_confidence(estimator, ce_input)
                    status = "completed"
            except Exception as error:
                logger.exception(
                    "Failed instance_id=%s model=%s\nProblem statement:\n%s",
                    item["instance_id"],
                    item["model"],
                    item["problem_statement"],
                )
                counts["failed"] += 1
                failures.append(
                    _failure_record(
                        item=item,
                        error=error,
                        estimator_cfg=estimator_cfg,
                        dataset_cfg=dataset_cfg,
                        output_dir=output_dir,
                    )
                )
                _write_failures(output_dir, failures)
                return None

            counts[status] += 1
            return cast(
                ConfEstimatedOpenHandsCEDataPoint,
                {
                    **item,
                    "estimated_confidence": ce_output.confidence,
                    "total_tokens": ce_output.total_tokens,
                    "generated_tokens": ce_output.generated_tokens,
                    "cost": ce_output.cost,
                },
            )

    results = await asyncio.gather(*(estimate_confidence_for_item(item) for item in dataset_items))
    _write_failures(output_dir, failures)
    logger.info(
        "Confidence-estimation report: skipped=%d completed=%d failed=%d",
        counts["skipped"],
        counts["completed"],
        counts["failed"],
    )
    return [result for result in results if result is not None]


async def estimate_dataset_confidence_async_batched(
    *,
    estimator_cfg: SupportedConfidenceEstimatorConfig,
    dataset_cfg: HuggingFaceDatasetConfig,
    output_dir: Path,
    max_workers: int,
    resume: bool = False,
) -> list[ConfEstimatedOpenHandsCEDataPoint]:
    """Estimate fixed-size async cohorts, letting estimators coordinate work within each cohort."""
    dataset: Dataset = dataset_cfg.load_dataset()
    dataset_items = cast(list[OpenHandsCEDataPoint], dataset.to_list())
    llm_limiters = build_llm_limiters(agent_configs_for_estimator(estimator_cfg))
    results: list[ConfEstimatedOpenHandsCEDataPoint | None] = [None] * len(dataset_items)
    failures: list[dict[str, Any]] = []
    counts = {"skipped": 0, "completed": 0, "failed": 0}

    def record_failure(index: int, item: OpenHandsCEDataPoint, error: Exception) -> None:
        logger.error(
            "Failed instance_id=%s model=%s\nProblem statement:\n%s",
            item["instance_id"],
            item["model"],
            item["problem_statement"],
            exc_info=(type(error), error, error.__traceback__),
        )
        counts["failed"] += 1
        failures.append(
            _failure_record(
                item=item,
                error=error,
                estimator_cfg=estimator_cfg,
                dataset_cfg=dataset_cfg,
                output_dir=output_dir,
            )
        )
        _write_failures(output_dir, failures)
        results[index] = None

    def record_result(
        index: int,
        item: OpenHandsCEDataPoint,
        ce_output: ConfEstimationOutput,
    ) -> None:
        results[index] = cast(
            ConfEstimatedOpenHandsCEDataPoint,
            {
                **item,
                "estimated_confidence": ce_output.confidence,
                "total_tokens": ce_output.total_tokens,
                "generated_tokens": ce_output.generated_tokens,
                "cost": ce_output.cost,
            },
        )

    for start_index in range(0, len(dataset_items), max_workers):
        batch_items = dataset_items[start_index : start_index + max_workers]
        estimator = build_openhands_estimator(estimator_cfg, llm_limiters=llm_limiters)
        pending: list[tuple[int, OpenHandsCEDataPoint, ConfEstimationInput]] = []
        for offset, item in enumerate(batch_items):
            index = start_index + offset
            try:
                item_output_dir = output_dir / item["instance_id"] / item["model"]
                ce_input = ConfEstimationInput(
                    conversation_archive_path=Path(dataset_cfg.base_path / item["conversation_archive_path"]),
                    output_dir=item_output_dir,
                    instance_id=item["instance_id"],
                    model=item["model"],
                    problem_statement=item["problem_statement"],
                    benchmark=item.get("benchmark"),
                    trajectory_type=item.get("trajectory_type"),
                )
                ce_output = await asyncio.to_thread(estimator.get_saved_output, item_output_dir) if resume else None
            except Exception as error:
                record_failure(index, item, error)
                continue
            if ce_output is not None:
                logger.info(
                    "Skipped instance_id=%s model=%s because a saved estimator output was found",
                    item["instance_id"],
                    item["model"],
                )
                counts["skipped"] += 1
                record_result(index, item, ce_output)
            else:
                pending.append((index, item, ce_input))

        if not pending:
            continue
        try:
            batch_outputs = await _aestimate_confidence_batch(estimator, [ce_input for _, _, ce_input in pending])
            if len(batch_outputs) != len(pending):
                raise ValueError(f"Batch estimator returned {len(batch_outputs)} outputs for {len(pending)} inputs")
        except Exception as error:
            for index, item, _ in pending:
                record_failure(index, item, error)
            continue

        for (index, item, _), ce_output in zip(pending, batch_outputs, strict=True):  # type: ignore
            if isinstance(ce_output, Exception):
                record_failure(index, item, ce_output)
                continue
            counts["completed"] += 1
            record_result(index, item, ce_output)  # type: ignore

    _write_failures(output_dir, failures)
    logger.info(
        "Confidence-estimation report: skipped=%d completed=%d failed=%d",
        counts["skipped"],
        counts["completed"],
        counts["failed"],
    )
    return [result for result in results if result is not None]


def main(cfg: ConfidenceEstimationRunConfig, *, resume: bool = False) -> None:
    assert cfg.output.output_dir, "no output dir!"
    _allow_qwen_3_8_reasoning_effort(cfg)
    if cfg.execution_mode == "async":
        results = asyncio.run(
            estimate_dataset_confidence_async(
                estimator_cfg=cfg.estimator,
                dataset_cfg=cfg.dataset,
                output_dir=cfg.output.output_dir,
                max_workers=cfg.max_workers,
                resume=resume,
            )
        )
    elif cfg.execution_mode == "async_batched":
        results = asyncio.run(
            estimate_dataset_confidence_async_batched(
                estimator_cfg=cfg.estimator,
                dataset_cfg=cfg.dataset,
                output_dir=cfg.output.output_dir,
                max_workers=cfg.max_workers,
                resume=resume,
            )
        )
    else:
        results = estimate_dataset_confidence(
            estimator_cfg=cfg.estimator,
            dataset_cfg=cfg.dataset,
            output_dir=cfg.output.output_dir,
            max_workers=cfg.max_workers,
            resume=resume,
        )
    df: pd.DataFrame = pd.DataFrame(results)
    df.to_csv(cfg.output.output_dir / "results.csv")

    # compute some metrics
    metrics: ConfMetrics = metrics_from_df(df)
    metrics_path: Path = cfg.output.output_dir / "metrics.json"
    metrics_path.write_text(metrics.model_dump_json(indent=4))
    logging.info("Saved metrics to %s.", metrics_path)
    logging.info("Metrics: %s", metrics.get_metric_log())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate confidence for a configured dataset run.")
    parser.add_argument("config", type=Path, help="Path to the run configuration YAML file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing estimator output artifacts",
    )
    return parser.parse_args(argv)


def cli() -> None:
    litellm.drop_params = True
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    cfg = load_run_config(args.config, ConfidenceEstimationRunConfig)
    main(cfg=cfg, resume=args.resume)


if __name__ == "__main__":
    cli()
