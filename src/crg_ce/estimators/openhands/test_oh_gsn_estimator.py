import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from crg_ce.estimators.base_estimator import ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.openhands.config import OHGSNEstimatorConfig
from crg_ce.estimators.openhands.oh_gsn_estimator import OHGSNConfidenceEstimator
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import (
    GSNGraphGeneratorConfig,
    GSNGraphPopulatorConfig,
)
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.utils.litellm_utils import LiteLLMCallStats

DUMMY_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000000")


def _cfg() -> OHGSNEstimatorConfig:
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


def _goal(name: str, *, confidence: float = -1) -> GSNGoalNode:
    return GSNGoalNode(
        confidence=confidence,
        goal_name=name,
        auditable_claim=f"{name} is achieved",
        reasoning=f"{name} is relevant",
    )


def _archive_path(tmp_path: Path) -> Path:
    path = tmp_path / "conversation.tar.gz"
    path.write_bytes(b"archive")
    return path


def _llm_stats(
    model: str,
    *,
    calls: int,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
) -> LiteLLMCallStats:
    stats = LiteLLMCallStats()
    stats.record_usage(
        model=model,
        calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=0,
        total_tokens=prompt_tokens + completion_tokens,
        cost=cost,
    )
    return stats


def _run_estimator_with_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    graph: ConfidenceGraph,
    problem_statement: str = "test-problem-statement",
) -> float:
    received_problem_statements: list[str | None] = []

    class FakeGenerator:
        llm_stats = LiteLLMCallStats()

        def generate_graph(self, state, events, problem_statement=None, benchmark=None):
            received_problem_statements.append(problem_statement)
            return graph

        def populate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            assert graph_arg is graph
            return graph_arg

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.load_conversation_state_and_events_from_archive",
        lambda path: (SimpleNamespace(id=DUMMY_CONVERSATION_ID, max_iterations=1), []),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_generator",
        lambda cfg, *, generator_log_path=None: FakeGenerator(),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_populator",
        lambda cfg, *, generator_log_path=None: FakeGenerator(),
    )
    estimator = OHGSNConfidenceEstimator(_cfg())
    ce_input = ConfEstimationInput(
        conversation_archive_path=_archive_path(tmp_path),
        output_dir=tmp_path,
        instance_id="test-instance",
        model="test-model",
        problem_statement=problem_statement,
    )
    confidence = estimator.estimate_confidence(ce_input).confidence
    assert received_problem_statements == [problem_statement]
    return confidence


def test_oh_gsn_estimator_returns_validated_goal_zero_confidence(monkeypatch, tmp_path: Path) -> None:
    # This verifies the GSN estimator dereferences goal_zero_node_id instead of assuming graph.nodes[0].
    goal_zero = _goal("overall", confidence=0.82)
    sub_goal = _goal("sub-goal", confidence=0.7)
    graph = ConfidenceGraph(
        nodes=[sub_goal, goal_zero],
        edges=[ConfidenceEdge(source=sub_goal.id, target=goal_zero.id, relationship_type="decomposes_from")],
        goal_zero_node_id=goal_zero.id,
    )

    assert _run_estimator_with_graph(monkeypatch, tmp_path, graph) == 0.82


def test_oh_gsn_estimator_passes_problem_statement_to_graph_generator(monkeypatch, tmp_path: Path) -> None:
    # This verifies ConfEstimationInput task context reaches graph construction.
    goal_zero = _goal("overall", confidence=0.82)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)

    assert _run_estimator_with_graph(monkeypatch, tmp_path, graph, "Fix the reported bug.") == 0.82


def test_oh_gsn_estimator_writes_parent_problem_log(monkeypatch, tmp_path: Path) -> None:
    # This verifies parent-side GSN estimator logs are scoped to the same per-problem directory as graph artifacts.
    goal_zero = _goal("overall", confidence=0.82)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)
    generator_log_paths: list[Path | None] = []

    class FakeGenerator:
        def __init__(self, generator_log_path: Path | None = None) -> None:
            self.generator_log_path = generator_log_path
            self.llm_stats = LiteLLMCallStats()

        def generate_graph(self, state, events, problem_statement=None, benchmark=None):
            generator_log_paths.append(self.generator_log_path)
            logging.getLogger("crg_ce.tests.oh_gsn_estimator").warning("parent generate marker")
            return graph

        def populate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            assert graph_arg is graph
            logging.getLogger("crg_ce.tests.oh_gsn_estimator").warning("parent populate marker")
            return graph_arg

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.load_conversation_state_and_events_from_archive",
        lambda path: (SimpleNamespace(id=DUMMY_CONVERSATION_ID, max_iterations=1), []),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_generator",
        lambda cfg, *, generator_log_path=None: FakeGenerator(generator_log_path),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_populator",
        lambda cfg, *, generator_log_path=None: FakeGenerator(generator_log_path),
    )

    estimator = OHGSNConfidenceEstimator(_cfg())
    ce_input = ConfEstimationInput(
        conversation_archive_path=_archive_path(tmp_path),
        output_dir=tmp_path,
        instance_id="test-instance",
        model="test-model",
        problem_statement="test-problem-statement",
    )
    (tmp_path / "openhands_gsn.log").write_text("stale log content")

    assert estimator.estimate_confidence(ce_input).confidence == 0.82
    assert generator_log_paths == [tmp_path / "openhands_gsn.log"]
    log_text = (tmp_path / "openhands_gsn.log").read_text()
    assert "stale log content" not in log_text
    assert "parent generate marker" in log_text
    assert "parent populate marker" in log_text


def test_oh_gsn_estimator_accumulates_generator_and_populator_usage(monkeypatch, tmp_path: Path) -> None:
    # This verifies output usage sums every graph-construction and confidence-population LLM call.
    goal_zero = _goal("overall", confidence=0.82)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)
    generator_stats = [
        _llm_stats("openai/generator-model", calls=2, prompt_tokens=160, completion_tokens=40, cost=0.02),
        _llm_stats("openai/populator-model", calls=3, prompt_tokens=240, completion_tokens=60, cost=0.03),
    ]

    class FakeGenerator:
        def __init__(self, llm_stats: LiteLLMCallStats) -> None:
            self.llm_stats = llm_stats

        def generate_graph(self, state, events, problem_statement=None, benchmark=None):
            return graph

        def populate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            assert graph_arg is graph
            return graph_arg

    generators = iter(FakeGenerator(stats) for stats in generator_stats)
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.load_conversation_state_and_events_from_archive",
        lambda path: (SimpleNamespace(id=DUMMY_CONVERSATION_ID, max_iterations=1), []),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_generator",
        lambda cfg, *, generator_log_path=None: next(generators),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_populator",
        lambda cfg, *, generator_log_path=None: next(generators),
    )
    estimator = OHGSNConfidenceEstimator(_cfg())

    output = estimator.estimate_confidence(
        ConfEstimationInput(
            conversation_archive_path=_archive_path(tmp_path),
            output_dir=tmp_path,
            instance_id="test-instance",
            model="test-model",
            problem_statement="test-problem-statement",
        )
    )

    assert output.confidence == 0.82
    assert output.total_tokens == 500
    assert output.generated_tokens == 100
    assert output.cost == 0.05
    assert output.usage_by_model["openai/generator-model"].calls == 2
    assert output.usage_by_model["openai/generator-model"].total_tokens == 200
    assert output.usage_by_model["openai/populator-model"].calls == 3
    assert output.usage_by_model["openai/populator-model"].total_tokens == 300
    assert ConfEstimationOutput.model_validate_json((tmp_path / "output.json").read_text()) == output


def test_async_oh_gsn_estimator_uses_configured_limiters_for_each_graph_phase(monkeypatch, tmp_path: Path) -> None:
    # This verifies the async GSN orchestration avoids synchronous graph methods and gives generation and population
    # their separately configured limiters, assuming archive loading remains dispatched off-loop.
    goal_zero = _goal("overall", confidence=-1)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)
    received_limiters: list[LLMCallLimiter | None] = []

    class FakeGenerator:
        llm_stats = LiteLLMCallStats()

        def generate_graph(self, state, events, problem_statement=None, benchmark=None):
            raise AssertionError("Synchronous generation must not be used")

        def populate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            raise AssertionError("Synchronous population must not be used")

        async def agenerate_graph(self, state, events, problem_statement=None, benchmark=None):
            assert problem_statement == "test-problem-statement"
            return graph

        async def apopulate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            graph_arg.nodes[0] = graph_arg.nodes[0].model_copy(update={"confidence": 0.82})
            return graph_arg

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.load_conversation_state_and_events_from_archive",
        lambda path: (SimpleNamespace(id=DUMMY_CONVERSATION_ID, max_iterations=1), []),
    )

    def fake_build(cfg, *, generator_log_path=None, llm_limiter=None):
        received_limiters.append(llm_limiter)
        return FakeGenerator()

    monkeypatch.setattr("crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_generator", fake_build)
    monkeypatch.setattr("crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_populator", fake_build)
    cfg = _cfg()
    cfg.graph_generator.agent.max_concurrent_llm_calls = 2
    cfg.graph_populator.agent.max_concurrent_llm_calls = 3
    generation_limiter = LLMCallLimiter(2)
    population_limiter = LLMCallLimiter(3)
    estimator = OHGSNConfidenceEstimator(
        cfg,
        llm_limiters={
            id(cfg.graph_generator.agent): generation_limiter,
            id(cfg.graph_populator.agent): population_limiter,
        },
    )
    output = asyncio.run(
        estimator.aestimate_confidence(
            ConfEstimationInput(
                conversation_archive_path=_archive_path(tmp_path),
                output_dir=tmp_path,
                instance_id="test-instance",
                model="test-model",
                problem_statement="test-problem-statement",
            )
        )
    )

    assert output.confidence == 0.82
    assert received_limiters == [generation_limiter, population_limiter]


def test_async_oh_gsn_batch_constructs_every_graph_before_population(monkeypatch, tmp_path: Path) -> None:
    # This verifies batch GSN estimation has exactly the intended LLM phases: all graph construction finishes before
    # any confidence population begins. It assumes each fake generator represents an independent problem context.
    generated_count = 0
    received_limiters: list[LLMCallLimiter | None] = []

    class FakeGenerator:
        llm_stats = LiteLLMCallStats()

        async def agenerate_graph(self, state, events, problem_statement=None, benchmark=None):
            nonlocal generated_count
            await asyncio.sleep(0)
            generated_count += 1
            goal_zero = _goal(problem_statement or "overall", confidence=-1)
            return ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)

        async def apopulate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            assert generated_count == 2
            graph_arg.nodes[0] = graph_arg.nodes[0].model_copy(update={"confidence": 0.82})
            return graph_arg

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.load_conversation_state_and_events_from_archive",
        lambda path: (SimpleNamespace(id=DUMMY_CONVERSATION_ID, max_iterations=1), []),
    )

    def fake_build(cfg, *, generator_log_path=None, llm_limiter=None):
        received_limiters.append(llm_limiter)
        return FakeGenerator()

    monkeypatch.setattr("crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_generator", fake_build)
    monkeypatch.setattr("crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_populator", fake_build)
    cfg = _cfg()
    limiter = LLMCallLimiter(2)
    estimator = OHGSNConfidenceEstimator(cfg, llm_limiter=limiter)
    ce_inputs = []
    for index in range(2):
        output_dir = tmp_path / f"output-{index}"
        archive_path = tmp_path / f"conversation-{index}.tar.gz"
        archive_path.write_bytes(b"archive")
        ce_inputs.append(
            ConfEstimationInput(
                conversation_archive_path=archive_path,
                output_dir=output_dir,
                instance_id=f"instance-{index}",
                model="test-model",
                problem_statement=f"problem-{index}",
            )
        )

    outputs = asyncio.run(estimator.aestimate_confidence_batch(ce_inputs))

    assert [output.confidence for output in outputs] == [0.82, 0.82]  # type: ignore
    assert generated_count == 2
    assert received_limiters == [limiter, limiter, limiter, limiter]


def test_async_oh_gsn_batch_populates_successful_graph_after_peer_timeout(monkeypatch, tmp_path: Path) -> None:
    # This verifies one graph-generation timeout remains item-local and successful peers still reach population. It
    # assumes problem statements uniquely identify the fake batch items.
    populated_problem_statements: list[str] = []

    class FakeGenerator:
        llm_stats = LiteLLMCallStats()

        async def agenerate_graph(self, state, events, problem_statement=None, benchmark=None):
            if problem_statement == "problem-0":
                raise TimeoutError("request timed out")
            goal_zero = _goal(problem_statement, confidence=-1)
            return ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)

        async def apopulate_graph_confidences(self, graph_arg, state, events, problem_statement=None):
            populated_problem_statements.append(graph_arg.nodes[0].goal_name)
            graph_arg.nodes[0] = graph_arg.nodes[0].model_copy(update={"confidence": 0.82})
            return graph_arg

    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.load_conversation_state_and_events_from_archive",
        lambda path: (SimpleNamespace(id=DUMMY_CONVERSATION_ID, max_iterations=1), []),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_generator",
        lambda cfg, *, generator_log_path=None, llm_limiter=None: FakeGenerator(),
    )
    monkeypatch.setattr(
        "crg_ce.estimators.openhands.oh_gsn_estimator.build_graph_populator",
        lambda cfg, *, generator_log_path=None, llm_limiter=None: FakeGenerator(),
    )
    cfg = _cfg()
    estimator = OHGSNConfidenceEstimator(cfg, llm_limiter=LLMCallLimiter(2))
    archive_path = _archive_path(tmp_path)
    ce_inputs = [
        ConfEstimationInput(
            conversation_archive_path=archive_path,
            output_dir=tmp_path / f"output-{index}",
            instance_id=f"instance-{index}",
            model="test-model",
            problem_statement=f"problem-{index}",
        )
        for index in range(2)
    ]

    outputs = asyncio.run(estimator.aestimate_confidence_batch(ce_inputs))

    assert isinstance(outputs[0], TimeoutError)
    assert outputs[1] == ConfEstimationOutput(confidence=0.82)
    assert populated_problem_statements == ["problem-1"]


def test_oh_gsn_estimator_prefers_saved_output(tmp_path: Path) -> None:
    # This verifies GSN resume uses output.json as the authoritative result when it is available.
    goal_zero = _goal("overall", confidence=0.82)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)
    (tmp_path / "graph.json").write_text(graph.model_dump_json(indent=2))
    (tmp_path / "output.json").write_text(
        ConfEstimationOutput(confidence=0.1, total_tokens=100, generated_tokens=20, cost=0.01).model_dump_json(indent=2)
    )
    estimator = OHGSNConfidenceEstimator(_cfg())

    output = estimator.get_saved_output(tmp_path)

    assert output == ConfEstimationOutput(confidence=0.1, total_tokens=100, generated_tokens=20, cost=0.01)


def test_oh_gsn_estimator_loads_legacy_graph_output_without_saved_usage(tmp_path: Path) -> None:
    # This verifies GSN resume supports older graph-only artifacts and leaves unavailable usage at its sentinel values.
    goal_zero = _goal("overall", confidence=0.82)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[], goal_zero_node_id=goal_zero.id)
    (tmp_path / "graph.json").write_text(graph.model_dump_json(indent=2))
    estimator = OHGSNConfidenceEstimator(_cfg())

    output = estimator.get_saved_output(tmp_path)

    assert output == ConfEstimationOutput(confidence=0.82)


def test_oh_gsn_estimator_rejects_missing_goal_zero_node_id(monkeypatch, tmp_path: Path) -> None:
    # This verifies GSN estimator inputs must explicitly identify the goal-zero node.
    goal_zero = _goal("overall", confidence=0.82)
    graph = ConfidenceGraph(nodes=[goal_zero], edges=[])

    with pytest.raises(ValueError, match="missing goal_zero_node_id"):
        _run_estimator_with_graph(monkeypatch, tmp_path, graph)


def test_oh_gsn_estimator_rejects_non_goal_zero_node_reference(monkeypatch, tmp_path: Path) -> None:
    # This verifies goal_zero_node_id must reference a GSN goal node, not arbitrary graph content.
    evidence = EvidenceNodeV2(
        confidence=0.5,
        evidence="evidence",
        step_numbers=[1],
        auditable_claim="Evidence happened.",
        contribution="Evidence contributes.",
    )
    sub_goal = _goal("sub-goal", confidence=0.7)
    graph = ConfidenceGraph(
        nodes=[evidence, sub_goal],
        edges=[ConfidenceEdge(source=sub_goal.id, target=evidence.id, relationship_type="decomposes_from")],
        goal_zero_node_id=evidence.id,
    )

    with pytest.raises(ValueError, match="goal_zero_node_id"):
        _run_estimator_with_graph(monkeypatch, tmp_path, graph)


def test_oh_gsn_estimator_rejects_goal_zero_with_outgoing_edges(monkeypatch, tmp_path: Path) -> None:
    # This verifies goal zero is terminal in the confidence dependency graph.
    goal_zero = _goal("overall", confidence=0.82)
    sub_goal = _goal("sub-goal", confidence=0.7)
    graph = ConfidenceGraph(
        nodes=[goal_zero, sub_goal],
        edges=[ConfidenceEdge(source=goal_zero.id, target=sub_goal.id, relationship_type="decomposes_from")],
        goal_zero_node_id=goal_zero.id,
    )

    with pytest.raises(ValueError, match="outgoing edges"):
        _run_estimator_with_graph(monkeypatch, tmp_path, graph)
