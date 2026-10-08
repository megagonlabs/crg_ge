from pathlib import Path
from typing import Protocol, cast

from openhands.sdk import Event, LocalWorkspace
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.tool import FinishTool

from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import (
    AgenticGraphConversation,
    DivideAndConquerInterpTool,
    DivideAndConquerTool,
    GatherEvidenceForAgenticGraphActionV1,
    GatherEvidenceForAgenticGraphToolV1,
    ParticularizeInterpTool,
    ParticularizeTool,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import GOAL_EDGE_LABEL_DESCRIPTIONS, get_edge_label_descriptions
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import BaseGSNGraphGenerator, GSNGraphGeneratorConfig
from crg_ce.graph.nodes import GSNGoalNode
from crg_ce.graph.utils import get_leaves
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.openhands.replay_agent import get_new_llm_stats
from crg_ce.utils.openhands import ConversationState, build_agent
from crg_ce.utils.openhands_trajectory import render_trajectory

AGENTIC_EVIDENCE_GATHERING_MAX_ITERATIONS = 1
AGENTIC_EVIDENCE_GATHERING_ATTEMPTS = 3


class _RunnableAgenticGraphConversation(Protocol):
    def run(self) -> None: ...

    async def arun(self) -> None: ...


class AgenticGSNGraphGenerator(BaseGSNGraphGenerator):
    """Generate a GSN graph and evidence agentically in one OpenHands conversation."""

    def __init__(
        self,
        cfg: GSNGraphGeneratorConfig,
        *,
        generator_log_path: Path | None = None,
        llm_limiter: LLMCallLimiter | None = None,
    ) -> None:
        super().__init__(cfg, generator_log_path=generator_log_path, llm_limiter=llm_limiter)

    @property
    def _uses_interp_tools(self) -> bool:
        return self.cfg.agent.tools_preset == "gsn_agentic_graph_construction_interp"

    @staticmethod
    def _filter_prompt_fields(
        included_fields: list[str],
        *,
        trajectory: str | None,
        problem_statement: str | None,
    ) -> dict[str, str | None]:
        available_fields = {
            "trajectory": trajectory,
            "problem_statement": problem_statement,
        }
        requested_fields = dict.fromkeys(included_fields)
        unknown_fields = sorted(set(requested_fields) - set(available_fields))
        if unknown_fields:
            raise ValueError(f"Unsupported included prompt fields: {unknown_fields}")
        return {field: available_fields[field] for field in requested_fields}

    def _validate_goal_construction(self, conversation: AgenticGraphConversation) -> None:
        if len(conversation.confidence_graph.nodes) != 1:
            return
        status = conversation.state.execution_status
        events = list(conversation.state.events)
        errors = [f"{event.code}: {event.detail}" for event in events if isinstance(event, ConversationErrorEvent)]
        # Keep enough of the final response/tool observations to diagnose a
        # tool-free finish or rejected call without dumping the input trajectory.
        recent_events = [event.model_dump_json()[:4000] for event in events[-5:] if event.source != "user"]
        self.logger.error(
            "Agentic goal construction produced no strategy-developed goals: "
            "status=%s, event_count=%s, errors=%s, recent_events=%s",
            status,
            len(events),
            errors,
            recent_events,
        )
        detail = f"status={status}"
        if errors:
            detail += f"; last_error={errors[-1][:1000]}"
        raise ValueError(f"Agentic graph contains no strategy-developed goals ({detail})")

    def _run_goal_construction(
        self,
        *,
        system_prompt: str,
        instruction: str,
        root_goal: GSNGoalNode,
    ) -> AgenticGraphConversation:
        tools_preset = (
            "gsn_agentic_graph_construction_interp" if self._uses_interp_tools else "gsn_agentic_graph_construction"
        )
        agent_config = self.cfg.agent.model_copy(update={"tools_preset": tools_preset})
        self.log_prompt(system_prompt)
        self.log_prompt(instruction)
        conversation = AgenticGraphConversation(
            agent=build_agent(agent_config, system_prompt=system_prompt),
            workspace=LocalWorkspace(working_dir=Path.cwd()),
            persistence_dir=None,
            visualizer=None,
            delete_on_close=False,
            max_iteration_per_run=self.cfg.max_steps,
        )
        conversation.confidence_graph = ConfidenceGraph(
            nodes=[root_goal],
            edges=[],
            goal_zero_node_id=root_goal.id,
        )
        conversation.send_message(instruction)  # type: ignore
        metrics_before_run = conversation.state.stats.get_combined_metrics()
        cast(_RunnableAgenticGraphConversation, conversation).run()
        self.record_llm_stats(get_new_llm_stats(conversation, metrics_before_run, agent_config.model_name))

        self._validate_goal_construction(conversation)
        return conversation

    async def _arun_goal_construction(
        self,
        *,
        system_prompt: str,
        instruction: str,
        root_goal: GSNGoalNode,
    ) -> AgenticGraphConversation:
        if self.llm_limiter is None:
            raise RuntimeError("Async graph generation requires an LLM call limiter")
        tools_preset = (
            "gsn_agentic_graph_construction_interp" if self._uses_interp_tools else "gsn_agentic_graph_construction"
        )
        agent_config = self.cfg.agent.model_copy(update={"tools_preset": tools_preset})
        self.log_prompt(system_prompt)
        self.log_prompt(instruction)
        conversation = AgenticGraphConversation(
            agent=build_agent(agent_config, system_prompt=system_prompt, llm_limiter=self.llm_limiter),
            workspace=LocalWorkspace(working_dir=Path.cwd()),
            persistence_dir=None,
            visualizer=None,
            delete_on_close=False,
            max_iteration_per_run=self.cfg.max_steps,
        )
        conversation.confidence_graph = ConfidenceGraph(nodes=[root_goal], edges=[], goal_zero_node_id=root_goal.id)
        conversation.send_message(instruction)  # type: ignore
        metrics_before_run = conversation.state.stats.get_combined_metrics()
        await cast(_RunnableAgenticGraphConversation, conversation).arun()
        self.record_llm_stats(get_new_llm_stats(conversation, metrics_before_run, agent_config.model_name))
        self._validate_goal_construction(conversation)
        return conversation

    def _gather_evidence_in_conversation(
        self,
        *,
        conversation: AgenticGraphConversation,
        instruction: str,
        agent_model_name: str,
    ) -> ConfidenceGraph:
        conversation.agent.add_runtime_tools(
            GatherEvidenceForAgenticGraphToolV1.create(evidence_edge_labels=self.cfg.prompts.evidence_edge_labels)
        )
        conversation.max_iteration_per_run = AGENTIC_EVIDENCE_GATHERING_MAX_ITERATIONS
        conversation.send_message(instruction)  # type: ignore
        for attempt_index in range(AGENTIC_EVIDENCE_GATHERING_ATTEMPTS):
            metrics_before_run = conversation.state.stats.get_combined_metrics()
            event_count_before_run = len(conversation.state.events)
            cast(_RunnableAgenticGraphConversation, conversation).run()
            self.record_llm_stats(get_new_llm_stats(conversation, metrics_before_run, agent_model_name))

            evidence_phase_events = list(conversation.state.events[event_count_before_run:])
            if any(
                isinstance(event, ActionEvent) and isinstance(event.action, GatherEvidenceForAgenticGraphActionV1)
                for event in evidence_phase_events
            ):
                return conversation.confidence_graph
            if not evidence_phase_events:
                raise ValueError(
                    "Evidence gathering cannot resume after the agent finishes without a tool call"
                )  # should never happen? there will always be events
            self.logger.warning(
                "Agentic evidence attempt %s did not call %s; resuming the conversation",
                attempt_index + 1,
                GatherEvidenceForAgenticGraphToolV1.name,
            )
        raise ValueError(
            f"Missing {GatherEvidenceForAgenticGraphToolV1.name} tool call after "
            f"{AGENTIC_EVIDENCE_GATHERING_ATTEMPTS} attempts"
        )

    async def _agather_evidence_in_conversation(
        self,
        *,
        conversation: AgenticGraphConversation,
        instruction: str,
        agent_model_name: str,
    ) -> ConfidenceGraph:
        conversation.agent.add_runtime_tools(
            GatherEvidenceForAgenticGraphToolV1.create(evidence_edge_labels=self.cfg.prompts.evidence_edge_labels)
        )
        conversation.max_iteration_per_run = AGENTIC_EVIDENCE_GATHERING_MAX_ITERATIONS
        conversation.send_message(instruction)  # type: ignore
        for attempt_index in range(AGENTIC_EVIDENCE_GATHERING_ATTEMPTS):
            metrics_before_run = conversation.state.stats.get_combined_metrics()
            event_count_before_run = len(conversation.state.events)
            await cast(_RunnableAgenticGraphConversation, conversation).arun()
            self.record_llm_stats(get_new_llm_stats(conversation, metrics_before_run, agent_model_name))
            evidence_phase_events = list(conversation.state.events[event_count_before_run:])
            if any(
                isinstance(event, ActionEvent) and isinstance(event.action, GatherEvidenceForAgenticGraphActionV1)
                for event in evidence_phase_events
            ):
                return conversation.confidence_graph
            if not evidence_phase_events:
                raise ValueError("Evidence gathering cannot resume after the agent finishes without a tool call")
            self.logger.warning(
                "Agentic evidence attempt %s did not call %s; resuming the conversation",
                attempt_index + 1,
                GatherEvidenceForAgenticGraphToolV1.name,
            )
        raise ValueError(
            f"Missing {GatherEvidenceForAgenticGraphToolV1.name} tool call after "
            f"{AGENTIC_EVIDENCE_GATHERING_ATTEMPTS} attempts"
        )

    def generate_graph(
        self,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
        benchmark: str | None = None,
    ) -> ConfidenceGraph:
        if self.cfg.prompts.skip_system_user_messages_in_trajectory and not problem_statement:
            raise ValueError("Skipping initial trajectory messages requires a non-empty problem_statement")
        root_goal = self.get_goal_zero_node()
        trajectory = render_trajectory(
            events,
            state=state,
            start_at_first_action_event=self.cfg.prompts.skip_system_user_messages_in_trajectory,
        )
        prompt_fields = self._filter_prompt_fields(
            self.cfg.prompts.decompose_goal_included_fields,
            trajectory=trajectory,
            problem_statement=problem_statement,
        )
        if not any(value and value.strip() for value in prompt_fields.values()):
            raise ValueError("Agentic graph construction requires trajectory, problem_statement, or both")

        edge_relationship_types: dict = {
            **get_edge_label_descriptions(self.cfg.prompts.evidence_edge_labels),
            **GOAL_EDGE_LABEL_DESCRIPTIONS,
        }
        system_prompt = self.cfg.prompts.system_prompt_instruction.render(
            overall_goal=root_goal.goal_name,
            overall_goal_auditable_claim=root_goal.auditable_claim,
            edge_relationship_types=edge_relationship_types,
            domain_success_criteria=self.cfg.prompts.render_domain_success_criteria(benchmark),
        ).strip()
        goal_instruction = self.cfg.prompts.goal_decompose_instruction.render(
            overall_goal=root_goal.goal_name,
            overall_goal_auditable_claim=root_goal.auditable_claim,
            target_goal=root_goal.goal_name,
            target_goal_auditable_claim=root_goal.auditable_claim,
            divide_and_conquer_function_name=(
                DivideAndConquerInterpTool.name if self._uses_interp_tools else DivideAndConquerTool.name
            ),
            particularize_function_name=(
                ParticularizeInterpTool.name if self._uses_interp_tools else ParticularizeTool.name
            ),
            finish_function_name=FinishTool.name,
            max_goal_decomposition_depth=self.cfg.max_steps,
            **prompt_fields,
        ).strip()
        conversation = self._run_goal_construction(
            system_prompt=system_prompt,
            instruction=goal_instruction,
            root_goal=root_goal,
        )
        constructed_graph = conversation.confidence_graph
        sub_goals = [node for node in get_leaves(constructed_graph) if isinstance(node, GSNGoalNode)]
        evidence_instruction = self.cfg.prompts.gather_evidence_instruction.render(
            sub_goals=sub_goals,
            gather_evidence_function_name=GatherEvidenceForAgenticGraphToolV1.name,
            evidence_gathering_max_iterations=AGENTIC_EVIDENCE_GATHERING_MAX_ITERATIONS,
        ).strip()
        graph = self._gather_evidence_in_conversation(
            conversation=conversation,
            instruction=evidence_instruction,
            agent_model_name=self.cfg.agent.model_name,
        )
        self.logger.info("Generated agentic GSN graph with %s nodes and %s edges", len(graph.nodes), len(graph.edges))
        return graph

    async def agenerate_graph(
        self,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
        benchmark: str | None = None,
    ) -> ConfidenceGraph:
        if self.cfg.prompts.skip_system_user_messages_in_trajectory and not problem_statement:
            raise ValueError("Skipping initial trajectory messages requires a non-empty problem_statement")
        root_goal = self.get_goal_zero_node()
        trajectory = render_trajectory(
            events,
            state=state,
            start_at_first_action_event=self.cfg.prompts.skip_system_user_messages_in_trajectory,
        )
        prompt_fields = self._filter_prompt_fields(
            self.cfg.prompts.decompose_goal_included_fields,
            trajectory=trajectory,
            problem_statement=problem_statement,
        )
        if not any(value and value.strip() for value in prompt_fields.values()):
            raise ValueError("Agentic graph construction requires trajectory, problem_statement, or both")
        edge_relationship_types: dict[str, str] = {}
        for relationship, description in get_edge_label_descriptions(self.cfg.prompts.evidence_edge_labels).items():
            edge_relationship_types[relationship] = description
        for goal_relationship, description in GOAL_EDGE_LABEL_DESCRIPTIONS.items():
            edge_relationship_types[goal_relationship] = description
        system_prompt = self.cfg.prompts.system_prompt_instruction.render(
            overall_goal=root_goal.goal_name,
            overall_goal_auditable_claim=root_goal.auditable_claim,
            edge_relationship_types=edge_relationship_types,
            domain_success_criteria=self.cfg.prompts.render_domain_success_criteria(benchmark),
        ).strip()
        goal_instruction = self.cfg.prompts.goal_decompose_instruction.render(
            overall_goal=root_goal.goal_name,
            overall_goal_auditable_claim=root_goal.auditable_claim,
            target_goal=root_goal.goal_name,
            target_goal_auditable_claim=root_goal.auditable_claim,
            divide_and_conquer_function_name=(
                DivideAndConquerInterpTool.name if self._uses_interp_tools else DivideAndConquerTool.name
            ),
            particularize_function_name=(
                ParticularizeInterpTool.name if self._uses_interp_tools else ParticularizeTool.name
            ),
            finish_function_name=FinishTool.name,
            max_goal_decomposition_depth=self.cfg.max_steps,
            **prompt_fields,
        ).strip()
        conversation = await self._arun_goal_construction(
            system_prompt=system_prompt, instruction=goal_instruction, root_goal=root_goal
        )
        sub_goals = [node for node in get_leaves(conversation.confidence_graph) if isinstance(node, GSNGoalNode)]
        evidence_instruction = self.cfg.prompts.gather_evidence_instruction.render(
            sub_goals=sub_goals,
            gather_evidence_function_name=GatherEvidenceForAgenticGraphToolV1.name,
            evidence_gathering_max_iterations=AGENTIC_EVIDENCE_GATHERING_MAX_ITERATIONS,
        ).strip()
        return await self._agather_evidence_in_conversation(
            conversation=conversation,
            instruction=evidence_instruction,
            agent_model_name=self.cfg.agent.model_name,
        )
