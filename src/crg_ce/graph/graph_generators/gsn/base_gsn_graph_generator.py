import abc
import asyncio
import logging
from pathlib import Path
from threading import Lock

from openhands.sdk import Event

from crg_ce.estimators.openhands.config import (
    CondenseConfig,
    GSNGraphGeneratorConfig,
    GSNGraphPopulatorConfig,
    GSNPromptConfig,
    SupportedGatherEvidenceMethod,
    SupportedGSNGraphGeneratorType,
    SupportedGSNGraphPopulatorType,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.nodes import GSNGoalNode
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.utils.litellm_utils import LiteLLMCallStats
from crg_ce.utils.openhands import ConversationState, get_logger
from crg_ce.utils.prompt_logging import log_prompt

__all__ = [
    "BaseGSNGraphComponent",
    "BaseGSNGraphGenerator",
    "BaseGSNGraphPopulator",
    "CondenseConfig",
    "GSNGraphGeneratorConfig",
    "GSNGraphPopulatorConfig",
    "GSNPromptConfig",
    "SupportedGatherEvidenceMethod",
    "SupportedGSNGraphGeneratorType",
    "SupportedGSNGraphPopulatorType",
]


class BaseGSNGraphComponent:
    """Shared logging and usage accounting for graph construction and population."""

    logger: logging.Logger

    def __init__(
        self,
        *,
        generator_log_path: Path | None = None,
        llm_limiter: LLMCallLimiter | None = None,
    ) -> None:
        self.logger = get_logger(self.__class__.__name__)
        self.generator_log_path = generator_log_path
        self.llm_stats = LiteLLMCallStats()
        self._llm_stats_lock = Lock()
        self.llm_limiter = llm_limiter

    def record_llm_stats(self, stats: LiteLLMCallStats) -> None:
        with self._llm_stats_lock:
            self.llm_stats.merge(stats)

    def log_prompt(self, prompt: str) -> None:
        if self.generator_log_path is not None:
            log_prompt(prompt, self.generator_log_path.parent)

    def log_messages(self, messages: list[dict[str, str]]) -> None:
        for message in messages:
            self.log_prompt(message["content"])


class BaseGSNGraphGenerator(BaseGSNGraphComponent, abc.ABC):
    """Interface for the retained agentic GSN graph-construction stage."""

    cfg: GSNGraphGeneratorConfig

    def __init__(self, cfg: GSNGraphGeneratorConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

    def get_goal_zero_node(self) -> GSNGoalNode:
        return GSNGoalNode(
            goal_name=self.cfg.prompts.goal_zero_task_name,
            auditable_claim=self.cfg.prompts.goal_zero_auditable_claim,
            reasoning=self.cfg.prompts.goal_zero_reasoning,
        )

    @abc.abstractmethod
    def generate_graph(
        self,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
        benchmark: str | None = None,
    ) -> ConfidenceGraph:
        """Build a GSN confidence graph from a completed agent trajectory."""
        raise NotImplementedError

    async def agenerate_graph(
        self,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
        benchmark: str | None = None,
    ) -> ConfidenceGraph:
        return await asyncio.to_thread(self.generate_graph, state, events, problem_statement, benchmark)

    def __call__(
        self,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
        benchmark: str | None = None,
    ) -> ConfidenceGraph:
        return self.generate_graph(state, events, problem_statement=problem_statement, benchmark=benchmark)


class BaseGSNGraphPopulator(BaseGSNGraphComponent, abc.ABC):
    """Interface for the retained LiteLLM confidence-population stage."""

    cfg: GSNGraphPopulatorConfig

    def __init__(self, cfg: GSNGraphPopulatorConfig, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg

    @abc.abstractmethod
    def populate_graph_confidences(
        self,
        graph: ConfidenceGraph,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
    ) -> ConfidenceGraph:
        """Populate confidence values in an existing GSN graph."""
        raise NotImplementedError

    async def apopulate_graph_confidences(
        self,
        graph: ConfidenceGraph,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
    ) -> ConfidenceGraph:
        return await asyncio.to_thread(self.populate_graph_confidences, graph, state, events, problem_statement)
