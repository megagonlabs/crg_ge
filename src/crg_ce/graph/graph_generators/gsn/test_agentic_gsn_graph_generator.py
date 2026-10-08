from types import SimpleNamespace

import pytest
import yaml
from openhands.sdk.event import ActionEvent, AgentErrorEvent
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.tool.builtins.finish import FinishAction

from crg_ce.estimators.openhands.config import GSNGraphGeneratorConfig
from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import (
    AgenticGoalSpec,
    DivideAndConquerAction,
    GatheredEvidenceCatalogItemAgenticV1,
    GatheredEvidenceEdgeAgenticV1,
    GatherEvidenceForAgenticGraphActionV1,
    ParticularizeAction,
    apply_agentic_evidence_action,
    apply_agentic_graph_action,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator import AgenticGSNGraphGenerator
from crg_ce.resources import read_resource
from crg_ce.utils.litellm_utils import LiteLLMCallStats


def _generator() -> AgenticGSNGraphGenerator:
    cfg_data = yaml.safe_load(read_resource("graph/graph_generators/gsn/test_data/dummy_cfg.yaml"))
    cfg_data["generator_type"] = "agentic"
    cfg_data["prompts"].update(
        {
            "system_prompt_instruction": "prompts/gsn/direct/agentic/system_prompt.j2",
            "goal_decompose_instruction": "prompts/gsn/direct/agentic/goal_decomposition.j2",
            "gather_evidence_instruction": "prompts/gsn/direct/agentic/evidence_gathering.j2",
        }
    )
    return AgenticGSNGraphGenerator(GSNGraphGeneratorConfig.model_validate(cfg_data))


def _event(action, tool_name: str, response_id: str = "response-1") -> ActionEvent:
    return ActionEvent(
        source="agent",
        thought=[],
        action=action,
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}",
        tool_call=MessageToolCall(
            id=f"call-{tool_name}",
            name=tool_name,
            arguments=action.model_dump_json(),
            origin="completion",
        ),
        llm_response_id=response_id,
    )


def test_goal_construction_uses_one_fresh_conversation_with_five_steps(monkeypatch) -> None:
    # This verifies construction sends one complete prompt to a fresh bounded conversation carrying the graph itself.
    generator = _generator()
    root = generator.get_goal_zero_node()
    captured: dict[str, object] = {}

    class FakeConversation:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs
            self.state = SimpleNamespace(
                events=[],
                stats=SimpleNamespace(get_combined_metrics=lambda: object()),
            )

        def send_message(self, instruction: str) -> None:
            captured["instruction"] = instruction

        def run(self) -> None:
            particularized = AgenticGoalSpec(
                goal_identifier="Concrete result",
                auditable_claim="The concrete result is correct.",
                reasoning="Task-aware success",
            )
            action = ParticularizeAction(
                target_goal_identifier=root.goal_name,
                target_auditable_claim=root.auditable_claim,
                particularized_goal=particularized,
            )
            self.confidence_graph: ConfidenceGraph = apply_agentic_graph_action(self.confidence_graph, action)
            self.state.events = [
                _event(action, "particularize"),
                _event(FinishAction(message="Graph complete"), "finish", response_id="response-2"),
            ]

    agent = object()
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator.AgenticGraphConversation",
        FakeConversation,
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator.build_agent",
        lambda config, *, system_prompt: captured.update(system_prompt=system_prompt) or agent,
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator.get_new_llm_stats",
        lambda conversation, baseline, model: LiteLLMCallStats(),
    )

    conversation = generator._run_goal_construction(
        system_prompt="graph system prompt",
        instruction="construct from trajectory",
        root_goal=root,
    )

    assert len(conversation.confidence_graph.nodes) == 2
    assert len(conversation.state.events) == 2
    assert captured["instruction"] == "construct from trajectory"
    assert captured["system_prompt"] == "graph system prompt"
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["agent"] is agent
    assert kwargs["max_iteration_per_run"] == 5
    assert kwargs["persistence_dir"] is None


def test_generate_graph_requires_problem_statement_when_skipping_initial_trajectory_messages() -> None:
    # This verifies phase one cannot omit the task after removing its initial user message from the trajectory.
    generator = _generator()
    generator.cfg.prompts.skip_system_user_messages_in_trajectory = True
    generator.cfg.prompts.decompose_goal_included_fields = ["trajectory", "problem_statement"]

    with pytest.raises(ValueError, match="non-empty problem_statement"):
        generator.generate_graph(state=None, events=[])  # type: ignore[arg-type]


def test_evidence_gathering_retries_one_iteration_after_agent_error(monkeypatch) -> None:
    # This verifies a one-step evidence run resumes from OpenHands' corrective error event without another user message.
    generator = _generator()
    generator.cfg.prompts.evidence_edge_labels = ["supports", "undermines"]
    root = generator.get_goal_zero_node()
    graph = apply_agentic_graph_action(
        ConfidenceGraph(
            nodes=[root],
            edges=[],
            goal_zero_node_id=root.id,
        ),
        ParticularizeAction(
            target_goal_identifier=root.goal_name,
            target_auditable_claim=root.auditable_claim,
            particularized_goal=AgenticGoalSpec(
                goal_identifier="Concrete result",
                auditable_claim="The concrete result is correct.",
                reasoning="Task-aware success",
            ),
        ),
    )
    evidence_action = GatherEvidenceForAgenticGraphActionV1(
        evidence_catalog=[
            GatheredEvidenceCatalogItemAgenticV1(
                evidence_key="result",
                evidence="The result is correct.",
                step_numbers=[1],
                auditable_claim="The concrete result is correct.",
                contribution="Direct evidence.",
            )
        ],
        evidence_edges=[
            GatheredEvidenceEdgeAgenticV1(
                evidence_key="result",
                target_goal_identifier="Concrete result",
                relationship_type="supports",
            )
        ],
    )

    class FakeConversation:
        def __init__(self) -> None:
            self.agent = SimpleNamespace(add_runtime_tools=lambda tools: captured.update(runtime_tools=tools))
            self.confidence_graph = graph
            self.max_iteration_per_run = 5
            self.run_count = 0
            self.state = SimpleNamespace(
                events=[],
                stats=SimpleNamespace(get_combined_metrics=lambda: object()),
            )

        def send_message(self, instruction: str) -> None:
            assert instruction == "gather evidence"

        def run(self) -> None:
            self.run_count += 1
            if self.run_count == 1:
                self.state.events.append(
                    AgentErrorEvent(
                        tool_name="gather_evidence_for_agentic_graph_v1",
                        tool_call_id="invalid",
                        error="bad",
                    )
                )
                return
            self.confidence_graph = apply_agentic_evidence_action(self.confidence_graph, evidence_action)
            self.state.events.append(_event(evidence_action, "gather_evidence_for_agentic_graph_v1"))

    conversation = FakeConversation()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator.get_new_llm_stats",
        lambda conversation, baseline, model: LiteLLMCallStats(),
    )

    result = generator._gather_evidence_in_conversation(
        conversation=conversation,  # type: ignore[arg-type]
        instruction="gather evidence",
        agent_model_name=generator.cfg.agent.model_name,
    )

    assert conversation.max_iteration_per_run == 1
    assert conversation.run_count == 2
    assert result == conversation.confidence_graph
    runtime_tools = captured["runtime_tools"]
    assert isinstance(runtime_tools, list)
    tool_schema = runtime_tools[0].action_type.model_json_schema()
    edge_schema = next(
        definition
        for name, definition in tool_schema["$defs"].items()
        if name.startswith("ConfiguredGatheredEvidenceEdgeAgenticV1_")
    )
    assert edge_schema["properties"]["relationship_type"]["enum"] == ["supports", "undermines"]


def test_generate_graph_continues_the_construction_conversation_for_evidence(monkeypatch) -> None:
    # This verifies phase two reuses the live construction conversation and its graph rather than a LiteLLM call.
    generator = _generator()
    root = generator.get_goal_zero_node()
    graph = apply_agentic_graph_action(
        ConfidenceGraph(nodes=[root], edges=[], goal_zero_node_id=root.id),
        ParticularizeAction(
            target_goal_identifier=root.goal_name,
            target_auditable_claim=root.auditable_claim,
            particularized_goal=AgenticGoalSpec(
                goal_identifier="Concrete result",
                auditable_claim="The concrete result is correct.",
                reasoning="Task-aware success",
            ),
        ),
    )
    graph = apply_agentic_graph_action(
        graph,
        DivideAndConquerAction(
            target_goal_identifier="Concrete result",
            target_auditable_claim="The concrete result is correct.",
            sub_goals=[
                AgenticGoalSpec(
                    goal_identifier="Behavior",
                    auditable_claim="The behavior is correct.",
                    reasoning="Required behavior",
                ),
                AgenticGoalSpec(
                    goal_identifier="Compatibility",
                    auditable_claim="Compatibility is preserved.",
                    reasoning="Required compatibility",
                ),
            ],
        ),
    )
    conversation = SimpleNamespace(confidence_graph=graph)
    captured: dict[str, object] = {}
    monkeypatch.setattr(generator, "get_goal_zero_node", lambda: root)
    monkeypatch.setattr(generator, "_run_goal_construction", lambda **kwargs: conversation)
    monkeypatch.setattr(
        generator,
        "_gather_evidence_in_conversation",
        lambda **kwargs: captured.update(kwargs) or graph,
    )
    monkeypatch.setattr(
        "crg_ce.graph.graph_generators.gsn.agentic_gsn_graph_generator.render_trajectory",
        lambda events, **kwargs: "trajectory",
    )

    generator.generate_graph(state=None, events=[])  # type: ignore[arg-type]

    assert captured["conversation"] is conversation
    assert captured["agent_model_name"] == generator.cfg.agent.model_name
    assert "gather_evidence_for_agentic_graph_v1" in captured["instruction"]  # type: ignore
