import json
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode
from crg_ce.graph.utils import bfs_predecessor_levels

ROOT_GOAL_NAME = "Agent's Overall Task"
GRAPH_LAYERS = ["0", "1", "2", "3+"]
GraphData = dict[str, Any]


def read_graph(graph_path: Path) -> GraphData:
    graph = json.loads(graph_path.read_text())
    if not isinstance(graph, dict):
        raise TypeError(f"Expected graph object in {graph_path}, got {type(graph).__name__}")

    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise TypeError(f"Expected graph nodes list in {graph_path}, got {type(nodes).__name__}")
    for node in nodes:
        if not isinstance(node, dict):
            raise TypeError(f"Expected graph node object in {graph_path}, got {type(node).__name__}")
        if "id" not in node:
            raise ValueError(f"Graph node in {graph_path} is missing id")
        if "kind" not in node:
            raise ValueError(f"Graph node {node['id']} in {graph_path} is missing kind")

    edges = graph.get("edges")
    if not isinstance(edges, list):
        raise TypeError(f"Expected graph edges list in {graph_path}, got {type(edges).__name__}")
    node_ids = {node["id"] for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError(f"Duplicate node ids in {graph_path}")
    for edge in edges:
        if not isinstance(edge, dict):
            raise TypeError(f"Expected graph edge object in {graph_path}, got {type(edge).__name__}")
        for field in ["source", "target", "relationship_type"]:
            if field not in edge:
                raise ValueError(f"Graph edge in {graph_path} is missing {field}")
        if edge["source"] not in node_ids:
            raise ValueError(f"Unknown edge source id in {graph_path}: {edge['source']}")
        if edge["target"] not in node_ids:
            raise ValueError(f"Unknown edge target id in {graph_path}: {edge['target']}")

    return graph


def get_goal_zero_node_id(graph: GraphData) -> str:
    goal_zero_node_id: str | None = graph.get("goal_zero_node_id")
    if goal_zero_node_id is not None:
        node_ids = {node["id"] for node in graph["nodes"]}
        if goal_zero_node_id not in node_ids:
            raise ValueError(f"Unknown goal_zero_node_id: {goal_zero_node_id}")
        return goal_zero_node_id

    roots = [
        node for node in graph["nodes"] if node["kind"] == "GSNGoalNode" and node.get("goal_name") == ROOT_GOAL_NAME
    ]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root goal named {ROOT_GOAL_NAME!r}, found {len(roots)}")
    root_node_id: str = roots[0]["id"]
    return root_node_id


def get_node_depths(graph: GraphData) -> dict[str, int]:
    """Return each node's shortest undirected distance from the root goal."""
    goal_zero_node_id = get_goal_zero_node_id(graph)

    neighbor_ids_by_node_id: dict[str, list[str]] = defaultdict(list)
    for edge in graph["edges"]:
        neighbor_ids_by_node_id[edge["source"]].append(edge["target"])
        neighbor_ids_by_node_id[edge["target"]].append(edge["source"])

    queue: deque[tuple[str, int]] = deque([(goal_zero_node_id, 0)])
    node_depths: dict[str, int] = {}
    while queue:
        node_id, depth = queue.popleft()
        if node_id in node_depths:
            continue
        node_depths[node_id] = depth
        queue.extend((neighbor_id, depth + 1) for neighbor_id in neighbor_ids_by_node_id[node_id])

    if len(node_depths) != len(graph["nodes"]):
        unreachable_node_count = len(graph["nodes"]) - len(node_depths)
        raise ValueError(f"Graph has {unreachable_node_count} nodes unreachable from goal_zero_node_id")

    return node_depths


def get_node_ids_by_layer(graph: GraphData) -> dict[str, list[str]]:
    node_ids_by_layer: dict[str, list[str]] = {layer: [] for layer in GRAPH_LAYERS}
    for node_id, depth in get_node_depths(graph).items():
        layer = "3+" if depth >= 3 else str(depth)
        node_ids_by_layer[layer].append(node_id)
    return node_ids_by_layer


def max_node_depth(graph: GraphData) -> int:
    """Return the greatest shortest undirected distance from the root goal."""
    node_depths = get_node_depths(graph)
    return max(node_depths.values())


def get_confidence(node: dict[str, Any]) -> float:
    confidence = node.get("confidence")
    if confidence is None or confidence == -1:
        return math.nan
    if not isinstance(confidence, int | float):
        raise TypeError(f"Expected numeric confidence for node {node['id']}, got {type(confidence).__name__}")
    if 0 <= confidence <= 1:
        return float(confidence)
    raise ValueError(f"Invalid confidence for node {node['id']}: {confidence}")


def average_confidence(nodes: list[dict[str, Any]]) -> float:
    confidences = [confidence for node in nodes if not math.isnan(confidence := get_confidence(node))]
    if not confidences:
        return math.nan
    return sum(confidences) / len(confidences)


def render_graph_claims_compact(graph: ConfidenceGraph) -> str:
    """Render node identifiers, claims, and parents in breadth-first order from the root."""
    if graph.goal_zero_node_id is None:
        raise ValueError("Graph is missing goal_zero_node_id")
    nodes_by_id = {node.id: node for node in graph.nodes}
    root = nodes_by_id[graph.goal_zero_node_id]
    ordered_nodes = [node for level in bfs_predecessor_levels(graph, root) for node in level]

    if len(ordered_nodes) != len(graph.nodes):
        unreachable_node_count = len(graph.nodes) - len(ordered_nodes)
        raise ValueError(f"Graph has {unreachable_node_count} nodes unreachable from goal_zero_node_id")

    identifier_by_node_id: dict[str, str] = {}
    for node in ordered_nodes:
        if not isinstance(node, (GSNGoalNode, EvidenceNodeV2)):
            raise TypeError(f"Unsupported graph node type for rendering: {type(node).__name__}")
        identifier_by_node_id[node.id] = node.goal_name if isinstance(node, GSNGoalNode) else node.id

    parent_ids_by_node_id: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        parent_ids_by_node_id[edge.source].append(edge.target)

    rendered_nodes: list[str] = []
    for node in ordered_nodes:
        parent_identifiers = [identifier_by_node_id[parent_id] for parent_id in parent_ids_by_node_id[node.id]]
        rendered_nodes.append(
            f"Node: {identifier_by_node_id[node.id]}\n"
            f"Claim: {getattr(node, 'auditable_claim', '<no claim>')}\n"
            f"Parents: {', '.join(parent_identifiers) if parent_identifiers else 'None'}"
        )
    return "\n\n".join(rendered_nodes)
