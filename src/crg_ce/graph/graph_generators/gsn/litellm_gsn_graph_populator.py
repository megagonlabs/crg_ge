import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import partial

from openhands.sdk import Event
from openhands.sdk.event import ACPToolCallEvent, ActionEvent
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
)

from crg_ce.estimators.base_estimator import rescale_confidence
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import (
    BaseGSNGraphPopulator,
    GSNGraphPopulatorConfig,
)
from crg_ce.graph.graph_generators.gsn.litellm_models import ConfidenceEstimateLiteLLM
from crg_ce.graph.nodes import CENode, EvidenceNodeV2, GSNGoalNode
from crg_ce.graph.utils import (
    aggregate_goal_confidences,
    aggregate_goal_confidences_product_interp_prior,
    aggregate_goal_confidences_product_interp_verbalized,
    bfs_predecessors,
    get_confidence_leaves,
    get_goal_leaves,
    get_leaves,
    get_predecessor_contexts,
    log_space_product,
)
from crg_ce.utils.litellm_utils import (
    LiteLLMCallStats,
    acomplete_structured,
    complete_structured,
)
from crg_ce.utils.openhands import ConversationState
from crg_ce.utils.openhands_trajectory import (
    condense_uncited_action_steps,
    get_active_branch_events,
    render_trajectory,
)


def _validate_raw_confidence(confidence: float, *, scale_min: float, scale_max: float) -> None:
    """Require an in-range whole-number confidence before it is rescaled."""
    if not scale_min <= confidence <= scale_max:
        raise ValueError(f"Confidence must be between {scale_min} and {scale_max}: {confidence}")
    if not confidence.is_integer():
        raise ValueError(f"Confidence must be a whole number: {confidence}")


class LiteGSNGraphPopulator(BaseGSNGraphPopulator):
    """Populate GSN confidence graphs with direct LiteLLM structured-output calls."""

    litellm_stats: LiteLLMCallStats

    def __init__(self, cfg: GSNGraphPopulatorConfig, **kwargs) -> None:
        super().__init__(cfg, **kwargs)
        self.litellm_stats = self.llm_stats

    def _confidence_trajectory_events(self, state: ConversationState, events: list[Event]) -> list[Event]:
        active_events = get_active_branch_events(events, state)
        if not self.cfg.prompts.skip_system_user_messages_in_trajectory:
            return active_events
        first_action_event_index = next(
            (index for index, event in enumerate(active_events) if isinstance(event, ACPToolCallEvent | ActionEvent)),
            None,
        )
        if first_action_event_index is None:
            raise ValueError("Cannot skip system and user messages from a trajectory without an action event")
        return active_events[first_action_event_index:]

    def _validate_confidence_problem_statement(self, problem_statement: str | None) -> None:
        if self.cfg.prompts.skip_system_user_messages_in_trajectory and (
            not problem_statement or not problem_statement.strip()
        ):
            raise ValueError("Skipping initial trajectory messages requires a non-empty problem_statement")

    def populate_graph_confidences(
        self,
        graph: ConfidenceGraph,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
    ) -> ConfidenceGraph:
        goal_zero = next(
            (node for node in graph.nodes if node.id == graph.goal_zero_node_id and isinstance(node, GSNGoalNode)),
            None,
        )
        if goal_zero is None:
            raise ValueError("Expected graph.goal_zero_node_id to identify a GSNGoalNode")

        system_prompt = self.cfg.prompts.confidence_estimation_system_prompt_instruction.render().strip()
        self._validate_confidence_problem_statement(problem_statement)

        def cited_step_numbers(node: CENode) -> set[int]:
            if isinstance(node, EvidenceNodeV2):
                return set(node.step_numbers)
            if self.cfg.condense.goal_confidence_context == "all_action_summaries":
                return set()
            return {
                step_number
                for dependent_node in bfs_predecessors(graph, node)
                for step_number in getattr(dependent_node, "step_numbers", [])
            }

        def render_confidence_trajectory(node: CENode) -> str:
            active_events = self._confidence_trajectory_events(state, events)
            if self.cfg.condense.mode == "summarize_uncited":
                condensed_events = condense_uncited_action_steps(active_events, cited_step_numbers(node))
                return render_trajectory(condensed_events, step_number_source_events=active_events)
            return render_trajectory(active_events)

        def assign_confidence(node: CENode, *, include_predecessor_contexts: bool = True) -> CENode:
            trajectory = render_confidence_trajectory(node)
            if isinstance(node, EvidenceNodeV2):
                instruction = self.cfg.prompts.assign_confidence_to_evidence_instruction.render(
                    evidence=node,
                    problem_statement=problem_statement,
                    scale_min=self.cfg.verbalization.scale_min,
                    scale_max=self.cfg.verbalization.scale_max,
                    scale_suffix=self.cfg.verbalization.scale_suffix,
                    target_goal=goal_zero.goal_name,
                    target_goal_auditable_claim=goal_zero.auditable_claim,
                    trajectory=trajectory,
                ).strip()
            elif isinstance(node, GSNGoalNode):
                instruction = self.cfg.prompts.assign_confidence_to_goal_instruction.render(
                    goal=node,
                    problem_statement=problem_statement,
                    predecessor_contexts=get_predecessor_contexts(graph, node) if include_predecessor_contexts else [],
                    scale_min=self.cfg.verbalization.scale_min,
                    scale_max=self.cfg.verbalization.scale_max,
                    scale_suffix=self.cfg.verbalization.scale_suffix,
                    overall_goal=goal_zero.goal_name,
                    overall_goal_auditable_claim=goal_zero.auditable_claim,
                    trajectory=trajectory,
                ).strip()
            else:
                raise ValueError(f"Unexpected confidence leaf node type: {type(node).__name__}")

            call_stats = LiteLLMCallStats()
            self.log_prompt(system_prompt)
            self.log_prompt(instruction)

            def complete_and_validate() -> ConfidenceEstimateLiteLLM:
                result = complete_structured(
                    model=self.cfg.agent.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": instruction},
                    ],
                    output_model=ConfidenceEstimateLiteLLM,
                    api_key=self.cfg.agent.api_key,
                    base_url=self.cfg.agent.api_base,
                    top_p=self.cfg.agent.top_p,
                    reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore[arg-type]
                    allowed_openai_params=self.cfg.agent.allowed_openai_params,
                    max_completion_tokens=self.cfg.agent.max_output_tokens,
                    stats=call_stats,
                )
                _validate_raw_confidence(
                    result.confidence,
                    scale_min=self.cfg.verbalization.scale_min,
                    scale_max=self.cfg.verbalization.scale_max,
                )
                return result

            retryer = Retrying(
                stop=stop_after_attempt(3),
                retry=retry_if_exception_type(ValueError),
                before_sleep=self._log_retry,
            )
            try:
                result = retryer(complete_and_validate)
            except RetryError as exc:
                self._log_failed_structured_output_attempt("confidence estimation", exc.last_attempt)
                raise ValueError(f"Missing valid confidence estimate for node_id={node.id} after 3 attempts") from exc
            finally:
                self.record_llm_stats(call_stats)

            confidence = rescale_confidence(
                min_score=self.cfg.verbalization.scale_min,
                max_score=self.cfg.verbalization.scale_max,
                score=result.confidence,
            )
            return node.model_copy(update={"confidence": confidence, "confidence_rationale": result.rationale})

        def populate_batch(nodes: list[CENode], *, include_predecessor_contexts: bool = True) -> None:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(
                        copy_context().run,
                        partial(
                            assign_confidence,
                            node,
                            include_predecessor_contexts=include_predecessor_contexts,
                        ),
                    )
                    for node in nodes
                ]
                updated_nodes = [future.result() for future in futures]
            updated_by_id = {node.id: node for node in updated_nodes}
            graph.nodes[:] = [updated_by_id.get(node.id, node) for node in graph.nodes]

        if self.cfg.confidence_population_mode == "goal_leaves_product":
            populate_batch([node for node in get_goal_leaves(graph) if node.confidence < 0])
            graph = aggregate_goal_confidences(graph, log_space_product, aggregation_type="product")
            unpopulated_goal_ids = [
                node.id for node in graph.nodes if isinstance(node, GSNGoalNode) and node.confidence < 0
            ]
            if unpopulated_goal_ids:
                raise ValueError(f"Could not resolve confidence dependencies for goal nodes: {unpopulated_goal_ids}")
            return graph
        elif self.cfg.confidence_population_mode == "product_interp_verbalized":
            goal_leaves = get_goal_leaves(graph)
            populate_batch([node for node in goal_leaves if node.confidence < 0])
            goal_leaf_ids = {node.id for node in goal_leaves}
            populate_batch(
                [node for node in graph.nodes if isinstance(node, GSNGoalNode) and node.id not in goal_leaf_ids],
                include_predecessor_contexts=False,
            )
            return aggregate_goal_confidences_product_interp_verbalized(graph)
        elif self.cfg.confidence_population_mode == "product_interp_prior":
            populate_batch([node for node in get_goal_leaves(graph) if node.confidence < 0])
            return aggregate_goal_confidences_product_interp_prior(graph, self.cfg.interpolation_prior)
        elif self.cfg.confidence_population_mode == "all_nodes":
            initial_leaves = [node for node in get_leaves(graph) if node.confidence < 0]
            populate_batch(initial_leaves)
            while ready_nodes := get_confidence_leaves(graph):
                populate_batch(ready_nodes)

            unpopulated_nodes = [node.id for node in graph.nodes if node.confidence < 0]
            if unpopulated_nodes:
                raise ValueError(f"Could not resolve confidence dependencies for nodes: {unpopulated_nodes}")
            return graph
        else:
            raise ValueError(f"Unsupported confidence population mode: {self.cfg.confidence_population_mode}")

    async def apopulate_graph_confidences(
        self,
        graph: ConfidenceGraph,
        state: ConversationState,
        events: list[Event],
        problem_statement: str | None = None,
    ) -> ConfidenceGraph:
        if self.llm_limiter is None:
            raise RuntimeError("Async graph population requires an LLM call limiter")
        llm_limiter = self.llm_limiter
        goal_zero = next(
            (node for node in graph.nodes if node.id == graph.goal_zero_node_id and isinstance(node, GSNGoalNode)),
            None,
        )
        if goal_zero is None:
            raise ValueError("Expected graph.goal_zero_node_id to identify a GSNGoalNode")
        system_prompt = self.cfg.prompts.confidence_estimation_system_prompt_instruction.render().strip()
        self._validate_confidence_problem_statement(problem_statement)

        def cited_step_numbers(node: CENode) -> set[int]:
            if isinstance(node, EvidenceNodeV2):
                return set(node.step_numbers)
            if self.cfg.condense.goal_confidence_context == "all_action_summaries":
                return set()
            return {
                step_number
                for dependent_node in bfs_predecessors(graph, node)
                for step_number in getattr(dependent_node, "step_numbers", [])
            }

        async def assign_confidence(node: CENode, *, include_predecessor_contexts: bool = True) -> CENode:
            active_events = self._confidence_trajectory_events(state, events)
            if self.cfg.condense.mode == "summarize_uncited":
                condensed_events = condense_uncited_action_steps(active_events, cited_step_numbers(node))
                trajectory = render_trajectory(condensed_events, step_number_source_events=active_events)
            else:
                trajectory = render_trajectory(active_events)
            if isinstance(node, EvidenceNodeV2):
                instruction = self.cfg.prompts.assign_confidence_to_evidence_instruction.render(
                    evidence=node,
                    problem_statement=problem_statement,
                    scale_min=self.cfg.verbalization.scale_min,
                    scale_max=self.cfg.verbalization.scale_max,
                    scale_suffix=self.cfg.verbalization.scale_suffix,
                    target_goal=goal_zero.goal_name,
                    target_goal_auditable_claim=goal_zero.auditable_claim,
                    trajectory=trajectory,
                ).strip()
            elif isinstance(node, GSNGoalNode):
                instruction = self.cfg.prompts.assign_confidence_to_goal_instruction.render(
                    goal=node,
                    problem_statement=problem_statement,
                    predecessor_contexts=get_predecessor_contexts(graph, node) if include_predecessor_contexts else [],
                    scale_min=self.cfg.verbalization.scale_min,
                    scale_max=self.cfg.verbalization.scale_max,
                    scale_suffix=self.cfg.verbalization.scale_suffix,
                    overall_goal=goal_zero.goal_name,
                    overall_goal_auditable_claim=goal_zero.auditable_claim,
                    trajectory=trajectory,
                ).strip()
            else:
                raise ValueError(f"Unexpected confidence leaf node type: {type(node).__name__}")

            call_stats = LiteLLMCallStats()
            self.log_prompt(system_prompt)
            self.log_prompt(instruction)
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError)
                ):
                    with attempt:
                        result = await acomplete_structured(
                            model=self.cfg.agent.model_name,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": instruction},
                            ],
                            output_model=ConfidenceEstimateLiteLLM,
                            llm_limiter=llm_limiter,
                            api_key=self.cfg.agent.api_key,
                            base_url=self.cfg.agent.api_base,
                            top_p=self.cfg.agent.top_p,
                            reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore[arg-type]
                            allowed_openai_params=self.cfg.agent.allowed_openai_params,
                            max_completion_tokens=self.cfg.agent.max_output_tokens,
                            stats=call_stats,
                        )
                        _validate_raw_confidence(
                            result.confidence,
                            scale_min=self.cfg.verbalization.scale_min,
                            scale_max=self.cfg.verbalization.scale_max,
                        )
            except RetryError as exc:
                raise ValueError(f"Missing valid confidence estimate for node_id={node.id} after 3 attempts") from exc
            finally:
                self.record_llm_stats(call_stats)
            confidence = rescale_confidence(
                min_score=self.cfg.verbalization.scale_min,
                max_score=self.cfg.verbalization.scale_max,
                score=result.confidence,
            )
            return node.model_copy(update={"confidence": confidence, "confidence_rationale": result.rationale})

        async def populate_batch(nodes: list[CENode], *, include_predecessor_contexts: bool = True) -> None:
            updated_nodes = await asyncio.gather(
                *(assign_confidence(node, include_predecessor_contexts=include_predecessor_contexts) for node in nodes)
            )
            updated_by_id = {node.id: node for node in updated_nodes}
            graph.nodes[:] = [updated_by_id.get(node.id, node) for node in graph.nodes]

        if self.cfg.confidence_population_mode == "goal_leaves_product":
            await populate_batch([node for node in get_goal_leaves(graph) if node.confidence < 0])
            graph = aggregate_goal_confidences(graph, log_space_product, aggregation_type="product")
            unpopulated_goal_ids = [
                node.id for node in graph.nodes if isinstance(node, GSNGoalNode) and node.confidence < 0
            ]
            if unpopulated_goal_ids:
                raise ValueError(f"Could not resolve confidence dependencies for goal nodes: {unpopulated_goal_ids}")
            return graph
        if self.cfg.confidence_population_mode == "product_interp_verbalized":
            goal_leaves = get_goal_leaves(graph)
            await populate_batch([node for node in goal_leaves if node.confidence < 0])
            goal_leaf_ids = {node.id for node in goal_leaves}
            await populate_batch(
                [node for node in graph.nodes if isinstance(node, GSNGoalNode) and node.id not in goal_leaf_ids],
                include_predecessor_contexts=False,
            )
            return aggregate_goal_confidences_product_interp_verbalized(graph)
        if self.cfg.confidence_population_mode == "product_interp_prior":
            await populate_batch([node for node in get_goal_leaves(graph) if node.confidence < 0])
            return aggregate_goal_confidences_product_interp_prior(graph, self.cfg.interpolation_prior)
        if self.cfg.confidence_population_mode == "all_nodes":
            await populate_batch([node for node in get_leaves(graph) if node.confidence < 0])
            while ready_nodes := get_confidence_leaves(graph):
                await populate_batch(ready_nodes)
            unpopulated_nodes = [node.id for node in graph.nodes if node.confidence < 0]
            if unpopulated_nodes:
                raise ValueError(f"Could not resolve confidence dependencies for nodes: {unpopulated_nodes}")
            return graph
        raise ValueError(f"Unsupported confidence population mode: {self.cfg.confidence_population_mode}")

    def _log_retry(self, retry_state: RetryCallState) -> None:
        if retry_state.outcome is None:
            return
        self._log_failed_structured_output_attempt(
            "structured-output",
            retry_state.outcome,
            attempt_number=retry_state.attempt_number,
            retrying=True,
        )

    def _log_failed_structured_output_attempt(
        self,
        operation: str,
        outcome: object,
        *,
        attempt_number: int | None = None,
        retrying: bool = False,
    ) -> None:
        exception = outcome.exception()  # type: ignore[attr-defined]
        content: object = None
        tool_calls: object = None
        if exception is not None:
            content = getattr(exception, "content", None)
            tool_calls = getattr(exception, "tool_calls", None)
        else:
            result = outcome.result()  # type: ignore[attr-defined]
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], list) and result[1]:
                response_message = result[1][-1]
                if isinstance(response_message, dict):
                    content = response_message.get("content")
                    tool_calls = response_message.get("tool_calls")

        action = "Retrying" if retrying else "Failed"
        self.logger.warning(
            "%s LiteLLM %s attempt %s: content=%r tool_calls=%r exception=%r",
            action,
            operation,
            attempt_number if attempt_number is not None else getattr(outcome, "attempt_number", None),
            content,
            tool_calls,
            exception,
        )
