import abc
import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, Field, FilePath

from crg_ce.utils.litellm_utils import LLMStats


def rescale_confidence(*, min_score: float, max_score: float, score: float) -> float:
    """Normalize an in-range score to a confidence in [0, 1]."""
    if not min_score <= score <= max_score:
        raise ValueError(f"Confidence must be between {min_score} and {max_score}: {score}")
    return (score - min_score) / (max_score - min_score)


def output_path_for_item(output_dir: Path, instance_id: str, model: str) -> Path:
    """Return the conventional output.json path for a dataset item."""
    return output_dir / instance_id / model / "output.json"


class ConfEstimationInput(BaseModel):
    conversation_archive_path: FilePath = Field(
        description="path to the saved OpenHands Conversation to predict confidence for"
    )
    output_dir: Path = Field(description="output directory for this given problem, for writing outputs (like graphs)")
    instance_id: str = Field(description="dataset instance identifier")
    model: str = Field(description="model that generated the instance trajectory")
    problem_statement: str = Field(
        description="user's task or problem statement, when available",
    )
    benchmark: str | None = Field(default=None, description="benchmark that supplied the task, when available")
    trajectory_type: str | None = Field(
        default=None,
        description="format of conversation_archive_path as supplied by the dataset, when available",
    )


class ModelUsage(BaseModel):
    calls: int = Field(ge=0)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)


def model_usage_from_stats(stats: LLMStats) -> dict[str, ModelUsage]:
    return {
        model: ModelUsage(
            calls=model_stats.calls,
            prompt_tokens=model_stats.prompt_tokens,
            completion_tokens=model_stats.completion_tokens,
            reasoning_tokens=model_stats.reasoning_tokens,
            total_tokens=model_stats.total_tokens,
            cost=model_stats.cost,
        )
        for model, model_stats in stats.by_model.items()
    }


class ConfEstimationOutput(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0, description="normalized confidence value")
    total_tokens: int = Field(default=-1, ge=-1, description="total input and generated tokens, or -1 if unavailable")
    generated_tokens: int = Field(
        default=-1,
        ge=-1,
        description="generated tokens including reasoning tokens, or -1 if unavailable",
    )
    cost: float = Field(default=-1, ge=-1, description="LLM cost in USD, or -1 if unavailable")
    usage_by_model: dict[str, ModelUsage] = Field(
        default_factory=dict,
        description="LLM call, token, and cost totals grouped by model",
    )


type ConfEstimationBatchItem = ConfEstimationOutput | Exception


class BaseConfidenceEstimator(abc.ABC):
    @abc.abstractmethod
    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        pass

    async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        """Estimate confidence without blocking the caller's event loop."""
        return await asyncio.to_thread(self.estimate_confidence, ce_input)

    async def aestimate_confidence_batch(
        self, ce_inputs: Sequence[ConfEstimationInput]
    ) -> Sequence[ConfEstimationBatchItem | BaseException]:
        """Estimate a batch, returning item-local exceptions without failing successful peers."""
        results = await asyncio.gather(
            *(self.aestimate_confidence(ce_input) for ce_input in ce_inputs),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        return list(results)

    def get_saved_output(self, item_output_dir: Path) -> ConfEstimationOutput | None:
        output_path = item_output_dir / "output.json"
        if not output_path.is_file():
            return None
        saved_output = ConfEstimationOutput.model_validate_json(output_path.read_text())
        if saved_output.confidence == -1:
            return None
        return saved_output

    def save_output(self, ce_output: ConfEstimationOutput, item_output_dir: Path) -> None:
        item_output_dir.mkdir(exist_ok=True, parents=True)
        save_path = item_output_dir / "output.json"
        save_path.write_text(ce_output.model_dump_json(indent=2))
