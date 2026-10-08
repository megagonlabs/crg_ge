import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import EvidenceEdgeLabel, GoalEdgeLabel
from crg_ce.graph.nodes import CENode, GSNGoalNode


@dataclass(frozen=True)
class PredecessorContext:
    node: CENode
    relationship_type: EvidenceEdgeLabel | GoalEdgeLabel | None


def outdated_aggregated_rationale(aggregation_type: str, rationale: str | None) -> str:
    """Mark a rationale obsolete after an aggregation replaces its confidence."""
    return f"*OUTDATED: aggregated: {aggregation_type}*: {rationale or ''}"


def get_leaves(graph: ConfidenceGraph) -> Iterator[CENode]:
    observed_targets: set[str] = {e.target for e in graph.edges}
    for node in graph.nodes:
        if node.id not in observed_targets:
            yield node


def get_goal_leaves(graph: ConfidenceGraph) -> list[GSNGoalNode]:
    nodes_by_id = {node.id: node for node in graph.nodes}
    goal_parent_ids = {edge.target for edge in graph.edges if isinstance(nodes_by_id[edge.source], GSNGoalNode)}
    return [node for node in graph.nodes if isinstance(node, GSNGoalNode) and node.id not in goal_parent_ids]


def log_space_product(values: list[float]) -> float:
    """Compute a product in log space, preserving the absorbing behavior of zero."""
    if any(value == 0 for value in values):
        return 0.0
    return math.exp(math.fsum(math.log(value) for value in values))


def aggregate_goal_confidences(
    graph: ConfidenceGraph,
    aggregate_confidences: Callable[[list[float]], float],
    *,
    aggregation_type: str,
) -> ConfidenceGraph:
    """Replace interior goal confidences bottom-up using their direct goal children."""
    nodes_by_id = {node.id: node for node in graph.nodes}
    goal_child_ids_by_parent_id: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        source = nodes_by_id[edge.source]
        target = nodes_by_id[edge.target]
        if isinstance(source, GSNGoalNode) and isinstance(target, GSNGoalNode):
            goal_child_ids_by_parent_id[target.id].append(source.id)

    goals_by_id = {node.id: node for node in graph.nodes if isinstance(node, GSNGoalNode)}
    resolved_goal_ids = set(goals_by_id) - set(goal_child_ids_by_parent_id)
    unresolved_goal_ids = set(goal_child_ids_by_parent_id)
    updated_goals_by_id = dict(goals_by_id)

    while unresolved_goal_ids:
        newly_resolved_goal_ids: set[str] = set()
        for goal_id in unresolved_goal_ids:
            child_ids = goal_child_ids_by_parent_id[goal_id]
            if not all(child_id in resolved_goal_ids for child_id in child_ids):
                continue
            child_confidences = [updated_goals_by_id[child_id].confidence for child_id in child_ids]
            if any(confidence < 0 for confidence in child_confidences):
                continue
            updated_goals_by_id[goal_id] = updated_goals_by_id[goal_id].model_copy(
                update={
                    "confidence": aggregate_confidences(child_confidences),
                    "confidence_rationale": outdated_aggregated_rationale(
                        aggregation_type,
                        updated_goals_by_id[goal_id].confidence_rationale,
                    ),
                }
            )
            newly_resolved_goal_ids.add(goal_id)

        if not newly_resolved_goal_ids:
            break
        resolved_goal_ids.update(newly_resolved_goal_ids)
        unresolved_goal_ids.difference_update(newly_resolved_goal_ids)

    return graph.model_copy(update={"nodes": [updated_goals_by_id.get(node.id, node) for node in graph.nodes]})


def aggregate_goal_confidences_product_interp_verbalized(graph: ConfidenceGraph) -> ConfidenceGraph:
    """Interpolate each parent's child product with its decomposition-independent verbal estimate."""
    nodes_by_id = {node.id: node for node in graph.nodes}
    goal_child_ids_by_parent_id: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        source = nodes_by_id[edge.source]
        target = nodes_by_id[edge.target]
        if isinstance(source, GSNGoalNode) and isinstance(target, GSNGoalNode):
            goal_child_ids_by_parent_id[target.id].append(source.id)

    goals_by_id = {node.id: node for node in graph.nodes if isinstance(node, GSNGoalNode)}
    if unset_goal_ids := [goal.id for goal in goals_by_id.values() if goal.confidence < 0]:
        raise ValueError(f"Cannot interpolate goals with unset verbalized confidence: {unset_goal_ids}")
    if unset_decomposition_goal_ids := [
        goal_id for goal_id in goal_child_ids_by_parent_id if goals_by_id[goal_id].confidence_in_children < 0
    ]:
        raise ValueError(f"Cannot interpolate goals with unset confidence_in_children: {unset_decomposition_goal_ids}")

    resolved_goal_ids = set(goals_by_id) - set(goal_child_ids_by_parent_id)
    unresolved_goal_ids = set(goal_child_ids_by_parent_id)
    updated_goals_by_id = dict(goals_by_id)
    while unresolved_goal_ids:
        newly_resolved_goal_ids: set[str] = set()
        for goal_id in unresolved_goal_ids:
            child_ids = goal_child_ids_by_parent_id[goal_id]
            if not all(child_id in resolved_goal_ids for child_id in child_ids):
                continue
            goal = updated_goals_by_id[goal_id]
            child_product = math.prod(updated_goals_by_id[child_id].confidence for child_id in child_ids)
            verbalized_confidence = goal.confidence
            decomposition_confidence = goal.confidence_in_children
            interpolated_confidence = (
                decomposition_confidence * child_product + (1 - decomposition_confidence) * verbalized_confidence
            )
            updated_goals_by_id[goal_id] = goal.model_copy(
                update={
                    "confidence": interpolated_confidence,
                    "confidence_rationale": (
                        f"*Interpolated: decomposition confidence={decomposition_confidence:g}, "
                        f"child product={child_product:g}, verbalized confidence={verbalized_confidence:g}*: "
                        f"{goal.confidence_rationale or ''}"
                    ),
                }
            )
            newly_resolved_goal_ids.add(goal_id)
        if not newly_resolved_goal_ids:
            raise ValueError(f"Could not resolve confidence dependencies for goal nodes: {sorted(unresolved_goal_ids)}")
        resolved_goal_ids.update(newly_resolved_goal_ids)
        unresolved_goal_ids.difference_update(newly_resolved_goal_ids)

    return graph.model_copy(update={"nodes": [updated_goals_by_id.get(node.id, node) for node in graph.nodes]})


def aggregate_goal_confidences_product_interp_prior(
    graph: ConfidenceGraph,
    prior: float,
) -> ConfidenceGraph:
    """Interpolate each parent's child product with a fixed prior confidence."""
    if not 0 <= prior <= 1:
        raise ValueError(f"Interpolation prior must be between 0 and 1: {prior}")

    nodes_by_id = {node.id: node for node in graph.nodes}
    goal_child_ids_by_parent_id: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        source = nodes_by_id[edge.source]
        target = nodes_by_id[edge.target]
        if isinstance(source, GSNGoalNode) and isinstance(target, GSNGoalNode):
            goal_child_ids_by_parent_id[target.id].append(source.id)

    goals_by_id = {node.id: node for node in graph.nodes if isinstance(node, GSNGoalNode)}
    goal_leaf_ids = set(goals_by_id) - set(goal_child_ids_by_parent_id)
    if unset_goal_leaf_ids := [goal_id for goal_id in goal_leaf_ids if goals_by_id[goal_id].confidence < 0]:
        raise ValueError(f"Cannot interpolate goal leaves with unset confidence: {unset_goal_leaf_ids}")
    if unset_decomposition_goal_ids := [
        goal_id for goal_id in goal_child_ids_by_parent_id if goals_by_id[goal_id].confidence_in_children < 0
    ]:
        raise ValueError(f"Cannot interpolate goals with unset confidence_in_children: {unset_decomposition_goal_ids}")

    resolved_goal_ids = set(goal_leaf_ids)
    unresolved_goal_ids = set(goal_child_ids_by_parent_id)
    updated_goals_by_id = dict(goals_by_id)
    while unresolved_goal_ids:
        newly_resolved_goal_ids: set[str] = set()
        for goal_id in unresolved_goal_ids:
            child_ids = goal_child_ids_by_parent_id[goal_id]
            if not all(child_id in resolved_goal_ids for child_id in child_ids):
                continue
            goal = updated_goals_by_id[goal_id]
            child_product = math.prod(updated_goals_by_id[child_id].confidence for child_id in child_ids)
            decomposition_confidence = goal.confidence_in_children
            interpolated_confidence = decomposition_confidence * child_product + (1 - decomposition_confidence) * prior
            updated_goals_by_id[goal_id] = goal.model_copy(
                update={
                    "confidence": interpolated_confidence,
                    "confidence_rationale": (
                        f"*Interpolated: decomposition confidence={decomposition_confidence:g}, "
                        f"child product={child_product:g}, prior={prior:g}*"
                    ),
                }
            )
            newly_resolved_goal_ids.add(goal_id)
        if not newly_resolved_goal_ids:
            raise ValueError(f"Could not resolve confidence dependencies for goal nodes: {sorted(unresolved_goal_ids)}")
        resolved_goal_ids.update(newly_resolved_goal_ids)
        unresolved_goal_ids.difference_update(newly_resolved_goal_ids)

    return graph.model_copy(update={"nodes": [updated_goals_by_id.get(node.id, node) for node in graph.nodes]})


def get_confidence_leaves(graph: ConfidenceGraph) -> list[CENode]:
    result: list[CENode] = []
    # dict: this node to its predecessors
    source_ids_by_target_id: dict[str, list[str]] = defaultdict(list)
    nodes_by_id: dict[str, CENode] = {node.id: node for node in graph.nodes}
    for edge in graph.edges:
        source_ids_by_target_id[edge.target].append(edge.source)
    # for each node in the graph:
    for node in graph.nodes:
        # if all predecessors have defined confidence: yield it
        if node.confidence == -1 and all(
            nodes_by_id[source_id].confidence >= 0.0 for source_id in source_ids_by_target_id[node.id]
        ):
            result.append(node)
    return result


def get_predecessor_contexts(graph: ConfidenceGraph, target_node: CENode) -> list[PredecessorContext]:
    nodes_by_id: dict[str, CENode] = {node.id: node for node in graph.nodes}
    if target_node.id not in nodes_by_id:
        raise ValueError(f"target_node.id={target_node.id} not in graph={graph}")

    return [
        PredecessorContext(
            node=nodes_by_id[edge.source],
            relationship_type=edge.relationship_type,
        )
        for edge in graph.edges
        if edge.target == target_node.id
    ]


def bfs_predecessors(graph: ConfidenceGraph, target_node: CENode) -> Iterator[CENode]:
    for level in bfs_predecessor_levels(graph, target_node):
        yield from level


def bfs_predecessor_levels(graph: ConfidenceGraph, target_node: CENode) -> Iterator[list[CENode]]:
    """Yield incoming graph nodes in breadth-first levels beginning with the target."""
    source_ids_by_target_id: dict[str, list[str]] = defaultdict(list)
    nodes_by_id: dict[str, CENode] = {node.id: node for node in graph.nodes}
    for edge in graph.edges:
        source_ids_by_target_id[edge.target].append(edge.source)
    if target_node.id not in nodes_by_id:
        raise ValueError(f"target_node.id={target_node.id} not in graph={graph}")

    queue: deque[list[str]] = deque([[target_node.id]])
    visited_node_ids: set[str] = set()
    while queue:
        level_node_ids = queue.popleft()
        level: list[CENode] = []
        next_level_node_ids: list[str] = []
        for node_id in level_node_ids:
            if node_id in visited_node_ids:
                continue
            visited_node_ids.add(node_id)
            level.append(nodes_by_id[node_id])
            next_level_node_ids.extend(source_ids_by_target_id[node_id])
        if level:
            yield level
        if next_level_node_ids:
            queue.append(next_level_node_ids)


def get_max_dependent_step_number(graph: ConfidenceGraph, node: CENode) -> int:
    step_numbers = [
        step_number
        for visited_node in bfs_predecessors(graph, node)
        for step_number in getattr(visited_node, "step_numbers", [])
    ]
    if not step_numbers:
        return -1
    return max(step_numbers)  # type: ignore
