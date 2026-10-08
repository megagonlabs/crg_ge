import yaml

from crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator import AgenticGSNGraphGenerator
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import (
    GSNGraphGeneratorConfig,
    GSNGraphPopulatorConfig,
)
from crg_ce.graph.graph_generators.gsn.builders import build_graph_generator, build_graph_populator
from crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator import LiteGSNGraphPopulator
from crg_ce.resources import read_resource


def _dummy_cfg() -> dict:
    return yaml.safe_load(read_resource("graph/graph_generators/gsn/test_data/dummy_cfg.yaml"))


def test_build_graph_generator_returns_agentic_generator() -> None:
    cfg = _dummy_cfg()
    cfg["generator_type"] = "agentic"
    assert isinstance(build_graph_generator(GSNGraphGeneratorConfig.model_validate(cfg)), AgenticGSNGraphGenerator)


def test_build_graph_populator_returns_litellm_populator() -> None:
    cfg = _dummy_cfg()
    cfg["generator_type"] = "litellm"
    assert isinstance(build_graph_populator(GSNGraphPopulatorConfig.model_validate(cfg)), LiteGSNGraphPopulator)
