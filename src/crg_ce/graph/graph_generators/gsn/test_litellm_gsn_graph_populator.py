import asyncio

import pytest
import yaml

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import GSNGraphPopulatorConfig
from crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator import LiteGSNGraphPopulator
from crg_ce.graph.graph_generators.gsn.litellm_models import ConfidenceEstimateLiteLLM
from crg_ce.graph.nodes.gsn.goal_node import EvidenceNodeV2, GSNGoalNode
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.resources import read_resource
from crg_ce.utils.general import resolve_template


@pytest.fixture
def dummy_populator() -> LiteGSNGraphPopulator:
    cfg_data = yaml.safe_load(read_resource("graph/graph_generators/gsn/test_data/dummy_cfg.yaml"))
    cfg_data["generator_type"] = "litellm"
    cfg_data["prompts"]["gather_evidence_method"] = "all_at_once"
    return LiteGSNGraphPopulator(cfg=GSNGraphPopulatorConfig.model_validate(cfg_data))


@pytest.mark.parametrize(
    "mode",
    ["all_nodes", "goal_leaves_product", "product_interp_verbalized", "product_interp_prior"],
)
def test_all_population_modes_remain_configurable(mode: str) -> None:
    cfg_data = yaml.safe_load(read_resource("graph/graph_generators/gsn/test_data/dummy_cfg.yaml"))
    cfg_data["generator_type"] = "litellm"
    cfg_data["confidence_population_mode"] = mode

    assert GSNGraphPopulatorConfig.model_validate(cfg_data).confidence_population_mode == mode


@pytest.mark.parametrize(
    ("goal_confidence_context", "expected_cited_steps"),
    [
        ("inherit_evidence_citations", [{38}, {38}, {38}]),
        ("all_action_summaries", [{38}, set(), set()]),
    ],
)
def test_populate_graph_confidences_uses_direct_structured_calls_in_dependency_order(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
    goal_confidence_context: str,
    expected_cited_steps: list[set[int]],
) -> None:
    # This verifies direct population scores evidence before its goal ancestors, propagates predecessor confidence,
    # and honors both summarized goal-context policies while assuming trajectory rendering itself is already tested.
    root = GSNGoalNode(goal_name="Successful result", auditable_claim="The task succeeded", reasoning="")
    leaf = GSNGoalNode(goal_name="Implementation works", auditable_claim="The implementation works", reasoning="")
    evidence = EvidenceNodeV2(
        evidence="Focused verification passed",
        step_numbers=[38],
        auditable_claim="The focused verification passed",
        contribution="Directly verifies the implementation",
    )
    graph = ConfidenceGraph(
        nodes=[root, leaf, evidence],
        edges=[
            ConfidenceEdge(source=evidence.id, target=leaf.id, relationship_type="supports"),
            ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )
    dummy_populator.cfg.condense.mode = "summarize_uncited"
    dummy_populator.cfg.condense.goal_confidence_context = goal_confidence_context  # type: ignore[assignment]
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_evidence_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_evidence.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )

    cited_steps: list[set[int]] = []
    prompts: list[str] = []
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )

    def fake_condense(events, cited_step_numbers):
        cited_steps.append(cited_step_numbers)
        return []

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.condense_uncited_action_steps",
        fake_condense,
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )

    def fake_complete_structured(**kwargs):
        assert kwargs["output_model"] is ConfidenceEstimateLiteLLM
        assert kwargs["messages"][0]["content"].startswith(
            "You estimate calibrated confidence for claims in a confidence-grounded assurance graph."
        )
        prompts.append(kwargs["messages"][1]["content"])
        prompt = prompts[-1]
        if "<target_evidence>" in prompt:
            return ConfidenceEstimateLiteLLM(confidence=8, rationale="direct evidence")
        if "Goal: Implementation works" in prompt:
            assert "Normalized Confidence: 0.80" in prompt
            return ConfidenceEstimateLiteLLM(confidence=7, rationale="supported leaf")
        assert "Goal: Successful result" in prompt
        assert "Normalized Confidence: 0.70" in prompt
        return ConfidenceEstimateLiteLLM(confidence=9, rationale="supported root")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.complete_structured",
        fake_complete_structured,
    )

    result = dummy_populator.populate_graph_confidences(graph, state=None, events=[])  # type: ignore[arg-type]

    assert cited_steps == expected_cited_steps
    assert {node.id: node.confidence for node in result.nodes} == {
        root.id: 0.9,
        leaf.id: 0.7,
        evidence.id: 0.8,
    }
    # This guarantees each structured LiteLLM assessment remains reviewable on its populated graph node.
    assert {node.id: node.confidence_rationale for node in result.nodes} == {
        root.id: "supported root",
        leaf.id: "supported leaf",
        evidence.id: "direct evidence",
    }


def test_populate_graph_confidences_scores_goal_leaves_then_aggregates_products(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies goal-leaf product mode makes no evidence or interior-goal LLM calls, omits unset evidence
    # confidence from leaf prompts, and deterministically propagates the product through the goal tree.
    root = GSNGoalNode(
        goal_name="Successful result",
        auditable_claim="The task succeeded",
        reasoning="",
        confidence_rationale="previous root assessment",
    )
    first_leaf = GSNGoalNode(goal_name="Implementation works", auditable_claim="The implementation works", reasoning="")
    second_leaf = GSNGoalNode(goal_name="Tests pass", auditable_claim="The tests pass", reasoning="")
    first_evidence = EvidenceNodeV2(
        evidence="Implementation inspection",
        step_numbers=[8],
        auditable_claim="The implementation is correct",
        contribution="Supports the implementation claim",
    )
    second_evidence = EvidenceNodeV2(
        evidence="Focused test output",
        step_numbers=[12],
        auditable_claim="Focused tests passed",
        contribution="Supports the test claim",
    )
    graph = ConfidenceGraph(
        nodes=[root, first_leaf, second_leaf, first_evidence, second_evidence],
        edges=[
            ConfidenceEdge(source=first_evidence.id, target=first_leaf.id, relationship_type="supports"),
            ConfidenceEdge(source=second_evidence.id, target=second_leaf.id, relationship_type="supports"),
            ConfidenceEdge(source=first_leaf.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second_leaf.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )
    dummy_populator.cfg.confidence_population_mode = "goal_leaves_product"
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )

    prompts: list[str] = []

    def fake_complete_structured(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        prompts.append(prompt)
        assert "<target_evidence>" not in prompt
        assert "Normalized Confidence:" not in prompt
        if "Goal: Implementation works" in prompt:
            return ConfidenceEstimateLiteLLM(confidence=8, rationale="implementation support")
        if "Goal: Tests pass" in prompt:
            return ConfidenceEstimateLiteLLM(confidence=5, rationale="test support")
        raise AssertionError(f"Unexpected confidence prompt: {prompt}")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.complete_structured",
        fake_complete_structured,
    )

    result = dummy_populator.populate_graph_confidences(graph, state=None, events=[])  # type: ignore[arg-type]
    confidences = {node.id: node.confidence for node in result.nodes}

    assert len(prompts) == 2
    assert confidences[root.id] == pytest.approx(0.4)
    assert confidences[first_leaf.id] == pytest.approx(0.8)
    assert confidences[second_leaf.id] == pytest.approx(0.5)
    assert confidences[first_evidence.id] == -1
    assert confidences[second_evidence.id] == -1
    result_nodes = {node.id: node for node in result.nodes}
    # This guarantees deterministic aggregation preserves, but visibly invalidates, a superseded node rationale.
    assert result_nodes[root.id].confidence_rationale == "*OUTDATED: aggregated: product*: previous root assessment"
    assert result_nodes[first_leaf.id].confidence_rationale == "implementation support"
    assert result_nodes[second_leaf.id].confidence_rationale == "test support"


def test_product_interp_verbalized_scores_parents_without_predecessor_claims(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies interpolation scores leaves normally, uses descendant citations only to summarize the provided
    # trajectory for parent prompts, omits predecessor claims, and recursively mixes resolved child confidences.
    root = GSNGoalNode(
        goal_name="Root",
        auditable_claim="The task succeeded",
        reasoning="",
        confidence_in_children=0.25,
    )
    middle = GSNGoalNode(
        goal_name="Middle",
        auditable_claim="The implementation is correct",
        reasoning="",
        confidence_in_children=0.5,
    )
    first = GSNGoalNode(goal_name="First", auditable_claim="First holds", reasoning="")
    second = GSNGoalNode(goal_name="Second", auditable_claim="Second holds", reasoning="")
    third = GSNGoalNode(goal_name="Third", auditable_claim="Third holds", reasoning="")
    evidence = EvidenceNodeV2(
        evidence="Generated assessment",
        step_numbers=[7],
        auditable_claim="A generated evidence claim",
        contribution="Supports First",
    )
    graph = ConfidenceGraph(
        nodes=[root, middle, first, second, third, evidence],
        edges=[
            ConfidenceEdge(source=evidence.id, target=first.id, relationship_type="supports"),
            ConfidenceEdge(source=first.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=third.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )
    dummy_populator.cfg.confidence_population_mode = "product_interp_verbalized"
    dummy_populator.cfg.condense.mode = "summarize_uncited"
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.condense_uncited_action_steps",
        lambda events, cited_steps: sorted(cited_steps),
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: f"provided trajectory evidence with cited steps {events}",
    )

    scores = {"First": 8, "Second": 5, "Third": 6, "Middle": 7, "Root": 9}
    parent_prompts: dict[str, str] = {}

    def fake_complete_structured(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        goal_name = next(name for name in scores if f"<target_goal>\nGoal: {name}\n" in prompt)
        if goal_name in {"Root", "Middle"}:
            parent_prompts[goal_name] = prompt
        return ConfidenceEstimateLiteLLM(confidence=scores[goal_name], rationale=f"direct {goal_name}")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.complete_structured",
        fake_complete_structured,
    )

    result = dummy_populator.populate_graph_confidences(graph, state=None, events=[])  # type: ignore[arg-type]
    nodes = {node.id: node for node in result.nodes}

    assert set(parent_prompts) == {"Root", "Middle"}
    assert all("No related claims are available" in prompt for prompt in parent_prompts.values())
    assert all("provided trajectory evidence with cited steps [7]" in prompt for prompt in parent_prompts.values())
    assert all("A generated evidence claim" not in prompt for prompt in parent_prompts.values())
    assert "First holds" not in parent_prompts["Middle"]
    assert "Second holds" not in parent_prompts["Middle"]
    assert "The implementation is correct" not in parent_prompts["Root"]
    assert "Third holds" not in parent_prompts["Root"]
    assert nodes[middle.id].confidence == pytest.approx(0.5 * (0.8 * 0.5) + 0.5 * 0.7)
    assert nodes[root.id].confidence == pytest.approx(0.25 * (0.55 * 0.6) + 0.75 * 0.9)
    assert nodes[first.id].confidence == pytest.approx(0.8)
    assert nodes[second.id].confidence == pytest.approx(0.5)
    assert nodes[third.id].confidence == pytest.approx(0.6)


def test_async_product_interp_verbalized_uses_the_same_mixture(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This guarantees asynchronous population applies the same parent-only estimate and interpolation equation as
    # synchronous population while executing structured calls through the shared LLM limiter.
    root = GSNGoalNode(
        goal_name="Root",
        auditable_claim="The task succeeded",
        reasoning="",
        confidence_in_children=0.4,
    )
    leaf = GSNGoalNode(goal_name="Leaf", auditable_claim="The requirement holds", reasoning="")
    graph = ConfidenceGraph(
        nodes=[root, leaf],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="particularizes")],
        goal_zero_node_id=root.id,
    )
    dummy_populator.cfg.confidence_population_mode = "product_interp_verbalized"
    dummy_populator.llm_limiter = LLMCallLimiter(1)
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "provided trajectory evidence",
    )

    async def fake_acomplete_structured(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        if "<target_goal>\nGoal: Root\n" in prompt:
            assert "The requirement holds" not in prompt
            return ConfidenceEstimateLiteLLM(confidence=9, rationale="direct root")
        return ConfidenceEstimateLiteLLM(confidence=5, rationale="leaf evidence")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.acomplete_structured",
        fake_acomplete_structured,
    )

    result = asyncio.run(dummy_populator.apopulate_graph_confidences(graph, state=None, events=[]))  # type: ignore[arg-type]
    nodes = {node.id: node for node in result.nodes}

    assert nodes[leaf.id].confidence == pytest.approx(0.5)
    assert nodes[root.id].confidence == pytest.approx(0.4 * 0.5 + 0.6 * 0.9)


def test_product_interp_prior_scores_only_leaves_and_uses_configured_prior(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies prior interpolation makes no parent LLM calls and recursively mixes each child product with the
    # configured fixed prior rather than a verbalized parent estimate.
    root = GSNGoalNode(
        goal_name="Root",
        auditable_claim="The task succeeded",
        reasoning="",
        confidence_in_children=0.25,
    )
    middle = GSNGoalNode(
        goal_name="Middle",
        auditable_claim="The implementation is correct",
        reasoning="",
        confidence_in_children=0.5,
    )
    first = GSNGoalNode(goal_name="First", auditable_claim="First holds", reasoning="")
    second = GSNGoalNode(goal_name="Second", auditable_claim="Second holds", reasoning="")
    third = GSNGoalNode(goal_name="Third", auditable_claim="Third holds", reasoning="")
    graph = ConfidenceGraph(
        nodes=[root, middle, first, second, third],
        edges=[
            ConfidenceEdge(source=first.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second.id, target=middle.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=middle.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=third.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )
    dummy_populator.cfg.confidence_population_mode = "product_interp_prior"
    dummy_populator.cfg.interpolation_prior = 0.2
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "provided trajectory evidence",
    )

    scores = {"First": 8, "Second": 5, "Third": 6}
    scored_goal_names: list[str] = []

    def fake_complete_structured(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        goal_name = next(name for name in scores if f"<target_goal>\nGoal: {name}\n" in prompt)
        scored_goal_names.append(goal_name)
        return ConfidenceEstimateLiteLLM(confidence=scores[goal_name], rationale=f"direct {goal_name}")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.complete_structured",
        fake_complete_structured,
    )

    result = dummy_populator.populate_graph_confidences(graph, state=None, events=[])  # type: ignore[arg-type]
    nodes = {node.id: node for node in result.nodes}

    assert set(scored_goal_names) == {"First", "Second", "Third"}
    assert nodes[middle.id].confidence == pytest.approx(0.5 * (0.8 * 0.5) + 0.5 * 0.2)
    assert nodes[root.id].confidence == pytest.approx(0.25 * (0.3 * 0.6) + 0.75 * 0.2)


def test_async_product_interp_prior_scores_only_goal_leaves(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This guarantees the async prior mode used by async run configs also avoids parent verbalization calls.
    root = GSNGoalNode(
        goal_name="Root",
        auditable_claim="The task succeeded",
        reasoning="",
        confidence_in_children=0.4,
    )
    leaf = GSNGoalNode(goal_name="Leaf", auditable_claim="The requirement holds", reasoning="")
    graph = ConfidenceGraph(
        nodes=[root, leaf],
        edges=[ConfidenceEdge(source=leaf.id, target=root.id, relationship_type="particularizes")],
        goal_zero_node_id=root.id,
    )
    dummy_populator.cfg.confidence_population_mode = "product_interp_prior"
    dummy_populator.cfg.interpolation_prior = 0.3
    dummy_populator.llm_limiter = LLMCallLimiter(1)
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "provided trajectory evidence",
    )

    async def fake_acomplete_structured(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        assert "<target_goal>\nGoal: Leaf\n" in prompt
        return ConfidenceEstimateLiteLLM(confidence=5, rationale="leaf evidence")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.acomplete_structured",
        fake_acomplete_structured,
    )

    result = asyncio.run(dummy_populator.apopulate_graph_confidences(graph, state=None, events=[]))  # type: ignore[arg-type]
    nodes = {node.id: node for node in result.nodes}

    assert nodes[leaf.id].confidence == pytest.approx(0.5)
    assert nodes[root.id].confidence == pytest.approx(0.4 * 0.5 + 0.6 * 0.3)


def test_async_goal_leaf_population_gathers_nodes_under_shared_llm_limit(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies async product population schedules independent goal leaves together, limits their LLM sections,
    # and retains deterministic product aggregation while leaving evidence confidences unpopulated.
    root = GSNGoalNode(goal_name="Root", auditable_claim="The task succeeded", reasoning="")
    first = GSNGoalNode(goal_name="First", auditable_claim="First succeeded", reasoning="")
    second = GSNGoalNode(goal_name="Second", auditable_claim="Second succeeded", reasoning="")
    evidence = EvidenceNodeV2(
        evidence="Verification",
        step_numbers=[],
        auditable_claim="Verification exists",
        contribution="Supports First",
    )
    graph = ConfidenceGraph(
        nodes=[root, first, second, evidence],
        edges=[
            ConfidenceEdge(source=evidence.id, target=first.id, relationship_type="supports"),
            ConfidenceEdge(source=first.id, target=root.id, relationship_type="decomposes_from"),
            ConfidenceEdge(source=second.id, target=root.id, relationship_type="decomposes_from"),
        ],
        goal_zero_node_id=root.id,
    )
    limiter = LLMCallLimiter(1)
    dummy_populator.llm_limiter = limiter
    dummy_populator.cfg.confidence_population_mode = "goal_leaves_product"
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    active = 0
    maximum = 0
    started: set[str] = set()

    async def fake_acomplete_structured(**kwargs):
        nonlocal active, maximum
        prompt = kwargs["messages"][1]["content"]
        started.add("First" if "Goal: First" in prompt else "Second")
        async with kwargs["llm_limiter"].slot():
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1
        confidence = 8 if "Goal: First" in prompt else 5
        return ConfidenceEstimateLiteLLM(confidence=confidence, rationale="scored")

    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.acomplete_structured",
        fake_acomplete_structured,
    )
    result = asyncio.run(dummy_populator.apopulate_graph_confidences(graph, state=None, events=[]))  # type: ignore[arg-type]
    confidences = {node.id: node.confidence for node in result.nodes}

    assert started == {"First", "Second"}
    assert maximum == 1
    assert confidences[root.id] == pytest.approx(0.4)
    assert confidences[evidence.id] == -1
    # This guarantees async LiteLLM population also persists the model rationale before aggregation replaces parents.
    result_nodes = {node.id: node for node in result.nodes}
    assert result_nodes[first.id].confidence_rationale == "scored"
    assert result_nodes[second.id].confidence_rationale == "scored"
    assert result_nodes[root.id].confidence_rationale == "*OUTDATED: aggregated: product*: "


def test_populate_graph_confidences_rejects_unknown_population_mode(
    dummy_populator: LiteGSNGraphPopulator,
) -> None:
    # This verifies runtime mutation or unvalidated construction cannot silently fall through to all-node population.
    root = GSNGoalNode(goal_name="Successful result", auditable_claim="The task succeeded", reasoning="")
    graph = ConfidenceGraph(nodes=[root], edges=[], goal_zero_node_id=root.id)
    dummy_populator.cfg.confidence_population_mode = "unknown"  # type: ignore[assignment]
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )

    with pytest.raises(ValueError, match="Unsupported confidence population mode: unknown"):
        dummy_populator.populate_graph_confidences(graph, state=None, events=[])  # type: ignore[arg-type]


def test_populate_graph_confidences_retries_non_whole_number_confidence(
    dummy_populator: LiteGSNGraphPopulator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies a decimal structured confidence is rejected and retried before graph confidence rescaling.
    root = GSNGoalNode(goal_name="Successful result", auditable_claim="The task succeeded", reasoning="")
    graph = ConfidenceGraph(nodes=[root], edges=[], goal_zero_node_id=root.id)
    dummy_populator.cfg.confidence_population_mode = "goal_leaves_product"
    dummy_populator.cfg.prompts.confidence_estimation_system_prompt_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/system_prompt.j2"
    )
    dummy_populator.cfg.prompts.assign_confidence_to_goal_instruction = resolve_template(
        "prompts/gsn/direct/agentic/estimation/assign_confidence_to_goal.j2"
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.get_active_branch_events",
        lambda events, state: [],
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.render_trajectory",
        lambda events, **kwargs: "rendered trajectory",
    )
    raw_confidences = iter((0.9, 8.0))
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.litellm_gsn_graph_populator.complete_structured",
        lambda **kwargs: ConfidenceEstimateLiteLLM(confidence=next(raw_confidences), rationale="scored"),
    )

    result = dummy_populator.populate_graph_confidences(graph, state=None, events=[])  # type: ignore[arg-type]

    assert result.nodes[0].confidence == pytest.approx(0.8)


