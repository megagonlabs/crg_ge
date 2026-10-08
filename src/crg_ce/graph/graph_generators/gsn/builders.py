from pathlib import Path

from crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator import AgenticGSNGraphGenerator
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import (
    BaseGSNGraphGenerator,
    BaseGSNGraphPopulator,
    GSNGraphGeneratorConfig,
    GSNGraphPopulatorConfig,
)
from crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator import LiteGSNGraphPopulator
from crg_ce.llm_concurrency import LLMCallLimiter


def build_graph_generator(
    cfg: GSNGraphGeneratorConfig,
    *,
    generator_log_path: Path | None = None,
    llm_limiter: LLMCallLimiter | None = None,
) -> BaseGSNGraphGenerator:
    return AgenticGSNGraphGenerator(cfg, generator_log_path=generator_log_path, llm_limiter=llm_limiter)


def build_graph_populator(
    cfg: GSNGraphPopulatorConfig,
    *,
    generator_log_path: Path | None = None,
    llm_limiter: LLMCallLimiter | None = None,
) -> BaseGSNGraphPopulator:
    return LiteGSNGraphPopulator(cfg, generator_log_path=generator_log_path, llm_limiter=llm_limiter)
