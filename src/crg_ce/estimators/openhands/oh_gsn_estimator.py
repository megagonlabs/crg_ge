import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import cast

from openhands.sdk import Event, get_logger
from scipy.stats import gmean

from crg_ce.estimators.base_estimator import (
    BaseConfidenceEstimator,
    ConfEstimationBatchItem,
    ConfEstimationInput,
    ConfEstimationOutput,
    model_usage_from_stats,
)
from crg_ce.estimators.calibration import temperature_scale_confidence
from crg_ce.estimators.openhands.config import (
    GSNPostHocAggregateConfig,
    OHGSNEstimatorConfig,
    output_dir_for_run_config,
)
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.graph_generators.gsn.base_gsn_graph_generator import (
    BaseGSNGraphComponent,
    BaseGSNGraphGenerator,
    BaseGSNGraphPopulator,
)
from crg_ce.graph.graph_generators.gsn.builders import build_graph_generator, build_graph_populator
from crg_ce.graph.nodes import GSNGoalNode
from crg_ce.graph.utils import (
    aggregate_goal_confidences,
    aggregate_goal_confidences_product_interp_prior,
    get_goal_leaves,
    log_space_product,
    outdated_aggregated_rationale,
)
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.utils.litellm_utils import LiteLLMCallStats
from crg_ce.utils.openhands import ConversationState, load_conversation_state_and_events_from_archive
from crg_ce.utils.problem_logging import problem_log_context


class GSNPostHocAggregateEstimator(BaseConfidenceEstimator):
    """Re-aggregate goal confidences in graphs produced by a completed GSN run."""

    def __init__(self, cfg: GSNPostHocAggregateConfig) -> None:
        self.cfg = cfg
        self.replay_output_dir = output_dir_for_run_config(cfg.replay_from)

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        source_graph_path = self.replay_output_dir / ce_input.instance_id / ce_input.model / "graph.json"
        if not source_graph_path.is_file():
            raise FileNotFoundError(
                f"GSN graph does not exist for instance_id={ce_input.instance_id!r}, "
                f"model={ce_input.model!r}: {source_graph_path}"
            )

        graph = ConfidenceGraph.model_validate_json(source_graph_path.read_text())
        if self.cfg.aggregation_type == "product":
            graph = aggregate_goal_confidences_with_product(graph)
        elif self.cfg.aggregation_type == "product_leaf_adaptive_claim_dropout":
            assert self.cfg.k_expected_leaves is not None
            graph = aggregate_goal_confidences_with_product_leaf_adaptive_claim_dropout(
                graph, self.cfg.k_expected_leaves
            )
        elif self.cfg.aggregation_type == "product_leaf_claim_dropout":
            assert self.cfg.d is not None
            graph = aggregate_goal_confidences_with_product_leaf_claim_dropout(
                graph,
                self.cfg.d,
                aggregation_type="product_leaf_claim_dropout",
            )
        elif self.cfg.aggregation_type == "product_leaf_temperature_scaled":
            assert self.cfg.temperature is not None
            graph = aggregate_goal_confidences_with_product_leaf_temperature_scaled(graph, self.cfg.temperature)
        elif self.cfg.aggregation_type == "product_all_temperature_scaled":
            assert self.cfg.temperature is not None
            graph = aggregate_goal_confidences_with_product_all_temperature_scaled(graph, self.cfg.temperature)
        elif self.cfg.aggregation_type == "temperature_scale_final":
            assert self.cfg.temperature is not None
            graph = aggregate_goal_confidences_with_temperature_scale_final(graph, self.cfg.temperature)
        elif self.cfg.aggregation_type == "geometric_mean":
            graph = aggregate_goal_confidences_with_geometric_mean(graph)
        elif self.cfg.aggregation_type == "geometric_mean_leaves":
            graph = aggregate_goal_confidences_with_geometric_mean_leaves(graph)
        elif self.cfg.aggregation_type == "arithmetic_mean":
            graph = aggregate_goal_confidences_with_arithmetic_mean(graph)
        elif self.cfg.aggregation_type == "minimum":
            graph = aggregate_goal_confidences_with_minimum(graph)
        elif self.cfg.aggregation_type == "maximum":
            graph = aggregate_goal_confidences_with_maximum(graph)
        elif self.cfg.aggregation_type == "union_bound":
            graph = aggregate_goal_confidences_with_union_bound(graph)
        elif self.cfg.aggregation_type == "simple_bp_goal_leaf_v1":
            from crg_ce.graph.goal_leaf_belief_propagation import (
                aggregate_goal_confidences_with_simple_bp_goal_leaf_v1,
            )

            graph = aggregate_goal_confidences_with_simple_bp_goal_leaf_v1(graph)
        elif self.cfg.aggregation_type == "simple_bp_goal_leaf_v2":
            from crg_ce.graph.goal_leaf_belief_propagation import (
                aggregate_goal_confidences_with_simple_bp_goal_leaf_v2,
            )

            graph = aggregate_goal_confidences_with_simple_bp_goal_leaf_v2(graph)
        elif self.cfg.aggregation_type == "product_interp_prior":
            assert self.cfg.interpolation_prior is not None
            graph = aggregate_goal_confidences_product_interp_prior(graph, self.cfg.interpolation_prior)
        else:
            raise ValueError(f"Unsupported GSN aggregation type: {self.cfg.aggregation_type}")

        goal_zero = _get_validated_goal_zero(graph)
        if goal_zero.confidence < 0:
            raise ValueError(f"Goal-zero confidence was not populated for node_id={goal_zero.id}")

        ce_input.output_dir.mkdir(exist_ok=True, parents=True)
        (ce_input.output_dir / "graph.json").write_text(graph.model_dump_json(indent=2))
        output = ConfEstimationOutput(confidence=goal_zero.confidence)
        self.save_output(output, ce_input.output_dir)
        return output


def aggregate_goal_confidences_with_product(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Replace non-leaf goal confidences with the product of their child-goal confidences."""
    return _aggregate_goal_confidences(graph, log_space_product, aggregation_type="product")


def claim_dropout_confidence(confidence: float, d: float) -> float:
    """Marginalize independent claim dropout for one confidence in closed form."""
    if not 0 <= confidence <= 1:
        raise ValueError(f"Expected confidence in [0, 1], got {confidence}")
    if not 0 <= d <= 1:
        raise ValueError(f"Expected d in [0, 1], got {d}")
    return d + (1 - d) * confidence


def aggregate_goal_confidences_with_product_leaf_claim_dropout(
    graph: ConfidenceGraph,
    d: float,
    *,
    aggregation_type: str,
) -> ConfidenceGraph:
    """Apply claim dropout once to goal leaves, then propagate their ordinary product."""
    goal_leaf_ids = {goal.id for goal in get_goal_leaves(graph)}
    dropout_graph = graph.model_copy(
        update={
            "nodes": [
                (
                    node.model_copy(
                        update={
                            "confidence": claim_dropout_confidence(node.confidence, d),
                            "confidence_rationale": outdated_aggregated_rationale(
                                aggregation_type, node.confidence_rationale
                            ),
                        }
                    )
                    if node.id in goal_leaf_ids
                    else node.model_copy(update={"confidence": -1})
                )
                if isinstance(node, GSNGoalNode)
                else node
                for node in graph.nodes
            ]
        }
    )
    return _aggregate_goal_confidences(
        dropout_graph,
        math.prod,
        aggregation_type=aggregation_type,
    )


def aggregate_goal_confidences_with_product_leaf_adaptive_claim_dropout(
    graph: ConfidenceGraph, k_expected_leaves: float
) -> ConfidenceGraph:
    """Choose dropout to retain k expected goal leaves, then apply leaf claim dropout."""
    if k_expected_leaves < 0:
        raise ValueError(f"Expected k_expected_leaves >= 0, got {k_expected_leaves}")
    n_goal_leaves = len(get_goal_leaves(graph))
    if n_goal_leaves == 0:
        raise ValueError("Cannot apply adaptive claim dropout to a graph with no goal leaves")
    d = max(0.0, 1.0 - k_expected_leaves / n_goal_leaves)
    return aggregate_goal_confidences_with_product_leaf_claim_dropout(
        graph,
        d,
        aggregation_type="product_leaf_adaptive_claim_dropout",
    )


def aggregate_goal_confidences_with_product_leaf_temperature_scaled(
    graph: ConfidenceGraph, temperature: float
) -> ConfidenceGraph:
    """Temperature-scale goal leaves, then propagate their product through the goal tree."""
    scaled_graph = _temperature_scale_goal_leaves_and_clear_internal_confidences(
        graph, temperature, aggregation_type="product_leaf_temperature_scaled"
    )
    return _aggregate_goal_confidences(
        scaled_graph,
        math.prod,
        aggregation_type="product_leaf_temperature_scaled",
    )


def aggregate_goal_confidences_with_product_all_temperature_scaled(
    graph: ConfidenceGraph, temperature: float
) -> ConfidenceGraph:
    """Temperature-scale each goal confidence before its product is propagated to its parent."""
    scaled_graph = _temperature_scale_goal_leaves_and_clear_internal_confidences(
        graph, temperature, aggregation_type="product_all_temperature_scaled"
    )
    return _aggregate_goal_confidences(
        scaled_graph,
        lambda confidences: temperature_scale_confidence(math.prod(confidences), temperature),
        aggregation_type="product_all_temperature_scaled",
    )


def aggregate_goal_confidences_with_temperature_scale_final(
    graph: ConfidenceGraph, temperature: float
) -> ConfidenceGraph:
    """Temperature-scale only the final root-goal confidence."""
    goal_zero = _get_validated_goal_zero(graph)
    return graph.model_copy(
        update={
            "nodes": [
                node.model_copy(
                    update={
                        "confidence": temperature_scale_confidence(node.confidence, temperature),
                        "confidence_rationale": outdated_aggregated_rationale(
                            "temperature_scale_final", node.confidence_rationale
                        ),
                    }
                )
                if node.id == goal_zero.id
                else node
                for node in graph.nodes
            ]
        }
    )


def _temperature_scale_goal_leaves_and_clear_internal_confidences(
    graph: ConfidenceGraph, temperature: float, *, aggregation_type: str
) -> ConfidenceGraph:
    """Retain only scaled goal-leaf confidences as inputs to a temperature-scaled aggregation."""
    goal_leaf_ids = {goal.id for goal in get_goal_leaves(graph)}
    return graph.model_copy(
        update={
            "nodes": [
                (
                    node.model_copy(
                        update={
                            "confidence": temperature_scale_confidence(node.confidence, temperature),
                            "confidence_rationale": outdated_aggregated_rationale(
                                aggregation_type, node.confidence_rationale
                            ),
                        }
                    )
                    if node.id in goal_leaf_ids
                    else node.model_copy(update={"confidence": -1})
                )
                if isinstance(node, GSNGoalNode)
                else node
                for node in graph.nodes
            ]
        }
    )


def aggregate_goal_confidences_with_geometric_mean(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Replace non-leaf goal confidences with the geometric mean of child-goal confidences."""

    return _aggregate_goal_confidences(graph, gmean, aggregation_type="geometric_mean")


def aggregate_goal_confidences_with_geometric_mean_leaves(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Set the root to the geometric mean of unchanged goal leaves and clear other internal goals."""
    goal_zero = _get_validated_goal_zero(graph)
    goal_leaves = get_goal_leaves(graph)
    if not goal_leaves:
        raise ValueError("Cannot aggregate a graph with no goal leaves")
    if any(goal.confidence < 0 for goal in goal_leaves):
        raise ValueError("Cannot aggregate goal leaves with unset confidence")

    goal_leaf_ids = {goal.id for goal in goal_leaves}
    root_confidence = float(gmean([goal.confidence for goal in goal_leaves]))
    return graph.model_copy(
        update={
            "nodes": [
                node
                if not isinstance(node, GSNGoalNode) or node.id in goal_leaf_ids
                else node.model_copy(
                    update={
                        "confidence": root_confidence if node.id == goal_zero.id else -1,
                        "confidence_rationale": outdated_aggregated_rationale(
                            "geometric_mean_leaves", node.confidence_rationale
                        ),
                    }
                )
                for node in graph.nodes
            ]
        }
    )


def aggregate_goal_confidences_with_arithmetic_mean(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Replace non-leaf goal confidences with the arithmetic mean of child-goal confidences."""
    return _aggregate_goal_confidences(graph, fmean, aggregation_type="arithmetic_mean")


def aggregate_goal_confidences_with_minimum(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Replace non-leaf goal confidences with the minimum child-goal confidence."""
    return _aggregate_goal_confidences(graph, min, aggregation_type="minimum")


def aggregate_goal_confidences_with_maximum(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Replace non-leaf goal confidences with the maximum child-goal confidence."""
    return _aggregate_goal_confidences(graph, max, aggregation_type="maximum")


def aggregate_goal_confidences_with_union_bound(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Lower-bound each conjunctive goal by subtracting its children's summed uncertainty."""
    return _aggregate_goal_confidences(
        graph,
        lambda confidences: max(0.0, 1.0 - sum(1.0 - confidence for confidence in confidences)),
        aggregation_type="union_bound",
    )


def _aggregate_goal_confidences(
    graph: ConfidenceGraph,
    aggregate_confidences: Callable[[list[float]], float],
    *,
    aggregation_type: str,
) -> ConfidenceGraph:
    return aggregate_goal_confidences(graph, aggregate_confidences, aggregation_type=aggregation_type)


@dataclass
class _AsyncGSNBatchItem:
    ce_input: ConfEstimationInput
    state: ConversationState | None
    events: list[Event]
    log_path: Path
    graph: ConfidenceGraph | None
    graph_generator: BaseGSNGraphGenerator
    graph_populator: BaseGSNGraphPopulator
    graph_components: list[BaseGSNGraphComponent]


def _load_events_for_confidence_input(ce_input: ConfEstimationInput) -> tuple[ConversationState | None, list[Event]]:
    if ce_input.trajectory_type is None:
        return load_conversation_state_and_events_from_archive(ce_input.conversation_archive_path)
    return load_conversation_state_and_events_from_archive(
        ce_input.conversation_archive_path,
        trajectory_type=ce_input.trajectory_type,
    )


class OHGSNConfidenceEstimator(BaseConfidenceEstimator):
    cfg: OHGSNEstimatorConfig

    def __init__(
        self,
        cfg: OHGSNEstimatorConfig,
        *,
        llm_limiter: LLMCallLimiter | None = None,
        llm_limiters: dict[int, LLMCallLimiter] | None = None,
    ) -> None:
        if llm_limiter is not None and llm_limiters is not None:
            raise ValueError("Specify either llm_limiter or llm_limiters, not both")
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.llm_limiter = llm_limiter
        self.llm_limiters = llm_limiters

    def _limiter_for(self, generator_cfg) -> LLMCallLimiter | None:
        if self.llm_limiters is None:
            return self.llm_limiter
        return self.llm_limiters[id(generator_cfg.agent)]

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        ce_input.output_dir.mkdir(exist_ok=True, parents=True)
        log_path: Path = ce_input.output_dir / "openhands_gsn.log"
        log_path.write_text("")
        with problem_log_context(log_path):
            state, events = _load_events_for_confidence_input(ce_input)
            self.logger.info("Constructing and populating graph for %s", ce_input.output_dir)
            graph_generator = build_graph_generator(self.cfg.graph_generator, generator_log_path=log_path)
            graph_populator = build_graph_populator(self.cfg.graph_populator, generator_log_path=log_path)
            graph = graph_generator.generate_graph(
                cast(ConversationState, state),
                events,
                problem_statement=ce_input.problem_statement,
                benchmark=ce_input.benchmark,
            )
            graph = graph_populator.populate_graph_confidences(
                graph,
                cast(ConversationState, state),
                events,
                problem_statement=ce_input.problem_statement,
            )
            goal_zero = _get_validated_goal_zero(graph)
            if goal_zero.confidence < 0:
                raise ValueError(f"Goal-zero confidence was not populated for node_id={goal_zero.id}")

            # save artifacts:
            graph_path: Path = ce_input.output_dir / "graph.json"
            graph_path.write_text(graph.model_dump_json(indent=2))
            self.logger.info(
                "Saved graph with %s nodes and final_confidence=%.3f for conversation %s to %s",
                len(graph.nodes),
                goal_zero.confidence,
                state.id if state is not None else ce_input.conversation_archive_path.stem,
                graph_path,
            )

            llm_stats = LiteLLMCallStats()
            for graph_component in (graph_generator, graph_populator):
                llm_stats.merge(graph_component.llm_stats)
            usage_available = llm_stats.prompt_tokens > 0 or llm_stats.completion_tokens > 0
            output = ConfEstimationOutput(
                confidence=goal_zero.confidence,
                total_tokens=llm_stats.total_tokens if usage_available else -1,
                generated_tokens=llm_stats.completion_tokens if usage_available else -1,
                cost=llm_stats.cost if llm_stats.cost > 0 else -1,
                usage_by_model=model_usage_from_stats(llm_stats),
            )
            self.save_output(output, ce_input.output_dir)
            return output

    async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        if self.llm_limiter is None and self.llm_limiters is None:
            return await super().aestimate_confidence(ce_input)
        ce_input.output_dir.mkdir(exist_ok=True, parents=True)
        log_path = ce_input.output_dir / "openhands_gsn.log"
        log_path.write_text("")
        with problem_log_context(log_path):
            state, events = await asyncio.to_thread(_load_events_for_confidence_input, ce_input)
            graph_generator = build_graph_generator(
                self.cfg.graph_generator,
                generator_log_path=log_path,
                llm_limiter=self._limiter_for(self.cfg.graph_generator),
            )
            graph_populator = build_graph_populator(
                self.cfg.graph_populator,
                generator_log_path=log_path,
                llm_limiter=self._limiter_for(self.cfg.graph_populator),
            )
            graph = await graph_generator.agenerate_graph(
                cast(ConversationState, state),
                events,
                problem_statement=ce_input.problem_statement,
                benchmark=ce_input.benchmark,
            )
            graph = await graph_populator.apopulate_graph_confidences(
                graph,
                cast(ConversationState, state),
                events,
                problem_statement=ce_input.problem_statement,
            )
            goal_zero = _get_validated_goal_zero(graph)
            if goal_zero.confidence < 0:
                raise ValueError(f"Goal-zero confidence was not populated for node_id={goal_zero.id}")

            graph_path = ce_input.output_dir / "graph.json"
            graph_path.write_text(graph.model_dump_json(indent=2))
            llm_stats = LiteLLMCallStats()
            for graph_component in (graph_generator, graph_populator):
                llm_stats.merge(graph_component.llm_stats)
            usage_available = llm_stats.prompt_tokens > 0 or llm_stats.completion_tokens > 0
            output = ConfEstimationOutput(
                confidence=goal_zero.confidence,
                total_tokens=llm_stats.total_tokens if usage_available else -1,
                generated_tokens=llm_stats.completion_tokens if usage_available else -1,
                cost=llm_stats.cost if llm_stats.cost > 0 else -1,
                usage_by_model=model_usage_from_stats(llm_stats),
            )
            self.save_output(output, ce_input.output_dir)
            return output

    async def aestimate_confidence_batch(
        self, ce_inputs: Sequence[ConfEstimationInput]
    ) -> Sequence[ConfEstimationBatchItem | BaseException]:
        if self.llm_limiter is None and self.llm_limiters is None:
            return await super().aestimate_confidence_batch(ce_inputs)

        def prepare(ce_input: ConfEstimationInput) -> _AsyncGSNBatchItem:
            ce_input.output_dir.mkdir(exist_ok=True, parents=True)
            log_path = ce_input.output_dir / "openhands_gsn.log"
            log_path.write_text("")
            state, events = _load_events_for_confidence_input(ce_input)
            graph_generator = build_graph_generator(
                self.cfg.graph_generator,
                generator_log_path=log_path,
                llm_limiter=self._limiter_for(self.cfg.graph_generator),
            )
            graph_populator = build_graph_populator(
                self.cfg.graph_populator,
                generator_log_path=log_path,
                llm_limiter=self._limiter_for(self.cfg.graph_populator),
            )
            return _AsyncGSNBatchItem(
                ce_input=ce_input,
                state=state,
                events=events,
                log_path=log_path,
                graph=None,
                graph_generator=graph_generator,
                graph_populator=graph_populator,
                graph_components=[graph_generator, graph_populator],
            )

        batch_items = [prepare(ce_input) for ce_input in ce_inputs]

        async def generate(batch_item: _AsyncGSNBatchItem) -> None:
            with problem_log_context(batch_item.log_path):
                batch_item.graph = await batch_item.graph_generator.agenerate_graph(
                    cast(ConversationState, batch_item.state),
                    batch_item.events,
                    problem_statement=batch_item.ce_input.problem_statement,
                    benchmark=batch_item.ce_input.benchmark,
                )

        generation_results = await asyncio.gather(
            *(generate(batch_item) for batch_item in batch_items),
            return_exceptions=True,
        )
        for generation_result in generation_results:
            if isinstance(generation_result, BaseException) and not isinstance(generation_result, Exception):
                raise generation_result

        async def populate(batch_item: _AsyncGSNBatchItem) -> None:
            assert batch_item.graph is not None
            with problem_log_context(batch_item.log_path):
                batch_item.graph = await batch_item.graph_populator.apopulate_graph_confidences(
                    batch_item.graph,
                    cast(ConversationState, batch_item.state),
                    batch_item.events,
                    problem_statement=batch_item.ce_input.problem_statement,
                )

        population_items = [
            batch_item
            for batch_item, generation_result in zip(batch_items, generation_results, strict=True)
            if not isinstance(generation_result, Exception)
        ]
        population_results = await asyncio.gather(
            *(populate(batch_item) for batch_item in population_items),
            return_exceptions=True,
        )
        for population_result in population_results:
            if isinstance(population_result, BaseException) and not isinstance(population_result, Exception):
                raise population_result
        population_results_by_id = {
            id(batch_item): result for batch_item, result in zip(population_items, population_results, strict=True)
        }

        outputs: list[ConfEstimationBatchItem] = []
        for batch_item, generation_result in zip(batch_items, generation_results, strict=True):
            if isinstance(generation_result, Exception):
                outputs.append(generation_result)
                continue
            population_result = population_results_by_id[id(batch_item)]
            if isinstance(population_result, Exception):
                outputs.append(population_result)
                continue
            try:
                assert batch_item.graph is not None
                goal_zero = _get_validated_goal_zero(batch_item.graph)
                if goal_zero.confidence < 0:
                    raise ValueError(f"Goal-zero confidence was not populated for node_id={goal_zero.id}")
                graph_path = batch_item.ce_input.output_dir / "graph.json"
                graph_path.write_text(batch_item.graph.model_dump_json(indent=2))
                llm_stats = LiteLLMCallStats()
                for graph_component in batch_item.graph_components:
                    llm_stats.merge(graph_component.llm_stats)
                usage_available = llm_stats.prompt_tokens > 0 or llm_stats.completion_tokens > 0
                output = ConfEstimationOutput(
                    confidence=goal_zero.confidence,
                    total_tokens=llm_stats.total_tokens if usage_available else -1,
                    generated_tokens=llm_stats.completion_tokens if usage_available else -1,
                    cost=llm_stats.cost if llm_stats.cost > 0 else -1,
                    usage_by_model=model_usage_from_stats(llm_stats),
                )
                self.save_output(output, batch_item.ce_input.output_dir)
                outputs.append(output)
            except Exception as error:
                outputs.append(error)
        return outputs

    def get_saved_output(self, item_output_dir: Path) -> ConfEstimationOutput | None:
        output_path = item_output_dir / "output.json"
        saved_output = (
            ConfEstimationOutput.model_validate_json(output_path.read_text()) if output_path.is_file() else None
        )
        graph_path = item_output_dir / "graph.json"
        if not graph_path.is_file():
            return saved_output

        graph = ConfidenceGraph.model_validate_json(graph_path.read_text())
        goal_zero = _get_validated_goal_zero(graph)
        if goal_zero.confidence == -1:
            return None
        if saved_output is None:
            # we have only graph.json to build a confidence result from, return using that
            return ConfEstimationOutput(confidence=goal_zero.confidence)
        return saved_output


def _get_validated_goal_zero(graph: ConfidenceGraph) -> GSNGoalNode:
    if graph.goal_zero_node_id is None:
        raise ValueError("GSN graph is missing goal_zero_node_id")

    nodes_by_id = {node.id: node for node in graph.nodes}
    goal_zero = nodes_by_id[graph.goal_zero_node_id]
    if not isinstance(goal_zero, GSNGoalNode):
        raise ValueError(f"Expected goal_zero_node_id to reference a GSNGoalNode, got {type(goal_zero)}")

    outgoing_edges = [edge for edge in graph.edges if edge.source == goal_zero.id]
    if outgoing_edges:
        raise ValueError(f"Goal-zero node has outgoing edges: {outgoing_edges}")

    return goal_zero
