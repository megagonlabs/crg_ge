"""Estimate confidence with a single LLM call over a verbalized GSN graph."""

import asyncio
from collections import defaultdict
from logging import Logger

from jinja2 import Template
from openhands.sdk import get_logger
from tenacity import AsyncRetrying, RetryError, Retrying, retry_if_exception_type, stop_after_attempt

from crg_ce.estimators.base_estimator import (
    BaseConfidenceEstimator,
    ConfEstimationInput,
    ConfEstimationOutput,
    model_usage_from_stats,
)
from crg_ce.estimators.openhands.config import (
    GSNGraphVerbalizationConfig,
    GSNNodeKind,
    GSNPlainVerbalizedEstimatorConfig,
)
from crg_ce.estimators.openhands.litellm_verbal_estimator import parse_confidence_percentage
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import GOAL_EDGE_LABEL_DESCRIPTIONS, ConfidenceEdge
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode
from crg_ce.graph.nodes.base_node import CENode
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.resources import read_resource
from crg_ce.utils.general import resolve_template
from crg_ce.utils.litellm_utils import LiteLLMCallStats, acomplete_text, complete_text
from crg_ce.utils.prompt_logging import log_prompt

_EVIDENCE_RELATIONSHIPS = {"proves", "supports", "refutes", "undermines", "unverified"}
_SUPPORTED_NODE_TYPES = (GSNGoalNode, EvidenceNodeV2)


def render_gsn_graph(graph: ConfidenceGraph, node_fields: dict[GSNNodeKind, list[str]]) -> str:
    """Render a GSN graph as a nested, claim-centric assurance case without identifiers."""
    node_fields = GSNGraphVerbalizationConfig(node_fields=node_fields).node_fields
    if graph.goal_zero_node_id is None:
        raise ValueError("GSN graph is missing goal_zero_node_id")

    nodes_by_id = {node.id: node for node in graph.nodes}
    if len(nodes_by_id) != len(graph.nodes):
        raise ValueError("GSN graph contains duplicate node ids")
    root = nodes_by_id[graph.goal_zero_node_id]
    if not isinstance(root, GSNGoalNode):
        raise TypeError(f"Expected the root node to be a GSNGoalNode, got {type(root).__name__}")
    if any(edge.source == root.id for edge in graph.edges):
        raise ValueError("The root goal must not have outgoing edges")

    unsupported_nodes = [node for node in graph.nodes if not isinstance(node, _SUPPORTED_NODE_TYPES)]
    if unsupported_nodes:
        raise TypeError(f"Unsupported GSN node type: {type(unsupported_nodes[0]).__name__}")

    incoming_edges: dict[str, list[ConfidenceEdge]] = defaultdict(list)
    for edge in graph.edges:
        incoming_edges[edge.target].append(edge)

    reached_node_ids: set[str] = set()
    active_node_ids: set[str] = set()

    def render_node(
        node: CENode,
        *,
        heading: str,
        indent: int,
    ) -> list[str]:
        if node.id in active_node_ids:
            raise ValueError("GSN graph contains a cycle")
        active_node_ids.add(node.id)
        reached_node_ids.add(node.id)

        prefix = " " * indent
        field_prefix = " " * (indent + 2)
        lines = [f"{prefix}{heading}"]
        if isinstance(node, GSNGoalNode):
            kind: GSNNodeKind = "GSNGoalNode"
            core_fields = [("Name", node.goal_name), ("Claim", node.auditable_claim)]
        elif isinstance(node, EvidenceNodeV2):
            kind = "EvidenceNodeV2"
            core_fields = [("Name", node.evidence), ("Claim", node.auditable_claim)]
        else:
            raise TypeError(f"Unsupported GSN node type: {type(node).__name__}")

        for label, value in core_fields:
            rendered_value = str(value) if value != "" else "(empty)"
            rendered_value = rendered_value.replace("\n", f"\n{field_prefix}  ")
            lines.append(f"{field_prefix}{label}: {rendered_value}")
        for field_name in node_fields.get(kind, []):
            value = getattr(node, field_name)
            if value == "" or value == []:
                rendered_value = "(empty)"
            elif isinstance(value, list):
                rendered_value = ", ".join(str(item) for item in value)
            else:
                rendered_value = str(value)
            rendered_value = rendered_value.replace("\n", f"\n{field_prefix}  ")
            label = field_name.replace("_", " ").title()
            lines.append(f"{field_prefix}{label}: {rendered_value}")

        node_incoming_edges = incoming_edges[node.id]
        if isinstance(node, GSNGoalNode):
            subgoal_edges: list[ConfidenceEdge] = []
            evidence_edges: list[ConfidenceEdge] = []
            for edge in node_incoming_edges:
                source = nodes_by_id[edge.source]
                if isinstance(source, GSNGoalNode):
                    if edge.relationship_type not in GOAL_EDGE_LABEL_DESCRIPTIONS:
                        raise ValueError("A goal-to-goal connection must use a goal relationship type")
                    subgoal_edges.append(edge)
                elif isinstance(source, EvidenceNodeV2):
                    if edge.relationship_type not in _EVIDENCE_RELATIONSHIPS:
                        raise ValueError("An evidence connection must use an evidence relationship type")
                    evidence_edges.append(edge)
                else:
                    raise TypeError(f"Unsupported predecessor type for a goal: {type(source).__name__}")

            for section_name, edges, child_heading in (
                ("SUBGOALS", subgoal_edges, "GOAL"),
                ("EVIDENCE", evidence_edges, "EVIDENCE"),
            ):
                if not edges:
                    continue
                lines.append(f"{field_prefix}{section_name}")
                for edge in edges:
                    source = nodes_by_id[edge.source]
                    lines.append(f"{field_prefix}- Connection: {edge.relationship_type}")
                    lines.extend(
                        render_node(
                            source,
                            heading=child_heading,
                            indent=indent + 4,
                        )
                    )
        elif node_incoming_edges:
            raise ValueError(f"Evidence nodes must not have incoming connections, got {len(node_incoming_edges)}")

        active_node_ids.remove(node.id)
        return lines

    rendered_lines = render_node(root, heading="ROOT GOAL", indent=0)
    unreachable_node_ids = set(nodes_by_id) - reached_node_ids
    if unreachable_node_ids:
        raise ValueError(f"GSN graph contains {len(unreachable_node_ids)} nodes unreachable from the root goal")
    return "\n".join(rendered_lines)


class GSNPlainVerbalizedEstimator(BaseConfidenceEstimator):
    """Estimate root-goal confidence directly from a loaded, text-rendered GSN graph."""

    cfg: GSNPlainVerbalizedEstimatorConfig
    logger: Logger
    instruction_template: Template
    ask_and_parse_output_instruction: str

    def __init__(self, cfg: GSNPlainVerbalizedEstimatorConfig, *, llm_limiter: LLMCallLimiter | None = None) -> None:
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.instruction_template = resolve_template(cfg.instruction_template)
        self.ask_and_parse_output_instruction = read_resource(cfg.ask_and_parse_output_instruction).strip()
        self.llm_limiter = llm_limiter

    def load_graph(self, ce_input: ConfEstimationInput) -> ConfidenceGraph:
        graph_path = self.cfg.load_graphs_from_path / ce_input.instance_id / ce_input.model / "graph.json"
        if not graph_path.is_file():
            raise FileNotFoundError(f"GSN graph does not exist: {graph_path}")
        self.logger.info("Loading graph from %s", graph_path)
        return ConfidenceGraph.model_validate_json(graph_path.read_text())

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        graph = self.load_graph(ce_input)
        rendered_graph = render_gsn_graph(graph, self.cfg.graph_verbalization.node_fields)
        instruction = self.instruction_template.render(
            graph=rendered_graph,
            output_instruction=self.ask_and_parse_output_instruction,
        )
        stats = LiteLLMCallStats()
        log_prompt(instruction, ce_input.output_dir)

        def complete_and_parse() -> float:
            response = complete_text(
                model=self.cfg.agent.model_name,
                messages=[{"role": "user", "content": instruction}],
                api_key=self.cfg.agent.api_key,
                base_url=self.cfg.agent.api_base,
                top_p=self.cfg.agent.top_p,
                reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore[arg-type]
                allowed_openai_params=self.cfg.agent.allowed_openai_params,
                max_completion_tokens=self.cfg.agent.max_output_tokens,
                stats=stats,
            )
            return parse_confidence_percentage(response)

        retryer = Retrying(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError))
        try:
            confidence = retryer(complete_and_parse)
        except RetryError as exc:
            raise ValueError("Could not parse GSN confidence percentage after 3 attempts") from exc

        output = ConfEstimationOutput(
            confidence=confidence,
            total_tokens=stats.total_tokens,
            generated_tokens=stats.completion_tokens,
            cost=stats.cost,
            usage_by_model=model_usage_from_stats(stats),
        )
        self.logger.info("Scored a plain verbalized GSN confidence of %s", output.confidence)
        self.save_output(output, ce_input.output_dir)
        return output

    async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        if self.llm_limiter is None:
            return await super().aestimate_confidence(ce_input)
        graph = await asyncio.to_thread(self.load_graph, ce_input)
        rendered_graph = render_gsn_graph(graph, self.cfg.graph_verbalization.node_fields)
        instruction = self.instruction_template.render(
            graph=rendered_graph,
            output_instruction=self.ask_and_parse_output_instruction,
        )
        stats = LiteLLMCallStats()
        log_prompt(instruction, ce_input.output_dir)
        try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError)):
                with attempt:
                    response = await acomplete_text(
                        model=self.cfg.agent.model_name,
                        messages=[{"role": "user", "content": instruction}],
                        llm_limiter=self.llm_limiter,
                        api_key=self.cfg.agent.api_key,
                        base_url=self.cfg.agent.api_base,
                        top_p=self.cfg.agent.top_p,
                        reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore[arg-type]
                        allowed_openai_params=self.cfg.agent.allowed_openai_params,
                        max_completion_tokens=self.cfg.agent.max_output_tokens,
                        stats=stats,
                    )
                    confidence = parse_confidence_percentage(response)
        except RetryError as exc:
            raise ValueError("Could not parse GSN confidence percentage after 3 attempts") from exc

        output = ConfEstimationOutput(
            confidence=confidence,
            total_tokens=stats.total_tokens,
            generated_tokens=stats.completion_tokens,
            cost=stats.cost,
            usage_by_model=model_usage_from_stats(stats),
        )
        self.save_output(output, ce_input.output_dir)
        return output
