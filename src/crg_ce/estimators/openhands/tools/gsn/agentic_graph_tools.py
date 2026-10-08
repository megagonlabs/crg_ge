import logging
from collections.abc import Sequence
from functools import cache
from typing import Annotated, Literal, Self, cast

from openhands.sdk import Action, Observation, TextContent, register_tool
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.tool import ToolDefinition, ToolExecutor
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.graph.edges import ConfidenceEdge, EvidenceEdgeLabel, GoalEdgeLabel
from crg_ce.graph.nodes import EvidenceNodeV2, GSNGoalNode
from crg_ce.graph.utils import get_leaves
from crg_ce.utils.graphs import render_graph_claims_compact

logger = logging.getLogger(__name__)


class AgenticGoalSpec(BaseModel):
    goal_identifier: str = Field(description="unique, short identifying key or name for this goal")
    auditable_claim: str = Field(description="positive, falsifiable success-condition claim")
    reasoning: str = Field(description="why this goal belongs in the argument")

    model_config = ConfigDict(extra="forbid")


class DivideAndConquerAction(Action):
    target_goal_identifier: str = Field(description="identifier or name of the existing goal to decompose")
    target_auditable_claim: str = Field(description="claim of the existing goal to decompose")
    sub_goals: Annotated[list[AgenticGoalSpec], Field(min_length=2)] = Field(
        description="two or more individually necessary and collectively sufficient conjuncts of the target claim"
    )


class ParticularizeAction(Action):
    target_goal_identifier: str = Field(description="identifier of the existing abstract goal to particularize")
    target_auditable_claim: str = Field(description="claim of the existing abstract goal to particularize")
    particularized_goal: AgenticGoalSpec = Field(
        description=(
            "a restatement of the abstract target claim into an equivalent claim more closely related to the "
            "specific task context"
        )
    )


class DivideAndConquerInterpAction(DivideAndConquerAction):
    confidence: float = Field(ge=0, le=1, description="confidence in the proposed decomposition, between 0 and 1")


class ParticularizeInterpAction(ParticularizeAction):
    confidence: float = Field(ge=0, le=1, description="confidence in the proposed particularization, between 0 and 1")


type AgenticGraphAction = DivideAndConquerAction | ParticularizeAction


class GatheredEvidenceCatalogItemAgenticV1(BaseModel):
    """Agentic-tool counterpart to the LiteLLM V3 evidence catalog item."""

    evidence_key: str = Field(description="unique key for this reusable evidence item within this response")
    evidence: str = Field(description="brief summary of this reusable piece of evidence")
    step_numbers: list[int] = Field(description="the step numbers which make up this piece of evidence")
    auditable_claim: str = Field(description="a restatement of this evidence as a falsifiable claim")
    contribution: str = Field(description="reasoning describing how this evidence contributes to the confidence graph")

    model_config = ConfigDict(extra="ignore")


class GatheredEvidenceEdgeAgenticV1(BaseModel):
    """V1 differs from LiteLLM V3 only by targeting the agentic goal identifier."""

    evidence_key: str = Field(description="key of the reusable evidence item from evidence_catalog")
    target_goal_identifier: str = Field(
        description="exact identifier of the leaf sub-goal this evidence contributes to"
    )
    relationship_type: EvidenceEdgeLabel = Field(
        description="relationship between this piece of evidence and the target goal's auditable claim"
    )

    model_config = ConfigDict(extra="ignore")


class GatherEvidenceForAgenticGraphActionV1(Action):
    """Structured evidence gathered by the retained agentic graph constructor."""

    evidence_catalog: list[GatheredEvidenceCatalogItemAgenticV1] = Field(
        description="reusable evidence items gathered from the trajectory"
    )
    evidence_edges: list[GatheredEvidenceEdgeAgenticV1] = Field(
        description="typed evidence relationships to sub-goals identified by exact goal identifier"
    )

    @model_validator(mode="after")
    def validate_response_references(self) -> Self:
        evidence_keys = [item.evidence_key for item in self.evidence_catalog]
        duplicate_evidence_keys = sorted({key for key in evidence_keys if evidence_keys.count(key) > 1})
        if duplicate_evidence_keys:
            raise ValueError(f"Duplicate evidence keys: {duplicate_evidence_keys}")

        catalog_keys = set(evidence_keys)
        edge_evidence_keys = {edge.evidence_key for edge in self.evidence_edges}
        unknown_evidence_keys = sorted(edge_evidence_keys - catalog_keys)
        if unknown_evidence_keys:
            raise ValueError(f"Unknown evidence keys: {unknown_evidence_keys}")

        edge_keys = [
            (edge.evidence_key, edge.target_goal_identifier, edge.relationship_type) for edge in self.evidence_edges
        ]
        duplicate_edge_keys = sorted({edge_key for edge_key in edge_keys if edge_keys.count(edge_key) > 1})
        if duplicate_edge_keys:
            raise ValueError(f"Duplicate evidence edges: {duplicate_edge_keys}")
        return self

    model_config = ConfigDict(extra="ignore")


def configured_gather_evidence_action_type(
    evidence_edge_labels: Sequence[EvidenceEdgeLabel],
) -> type[GatherEvidenceForAgenticGraphActionV1]:
    """Return the evidence-tool action schema for one configured set of evidence edge labels."""
    labels = tuple(evidence_edge_labels)
    if not labels:
        raise ValueError("Agentic evidence tool requires at least one evidence edge label")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Duplicate agentic evidence edge labels: {labels}")
    return _configured_gather_evidence_action_type(labels)


@cache
def _configured_gather_evidence_action_type(
    labels: tuple[EvidenceEdgeLabel, ...],
) -> type[GatherEvidenceForAgenticGraphActionV1]:
    """Create and cache one dynamic Pydantic action class per label tuple."""

    allowed_labels = Literal[labels]  # type: ignore[valid-type]  # Runtime Pydantic schema from configured labels.
    label_suffix = "_".join(labels)
    # Preserve the base edge fields while replacing its unrestricted relationship_type with this run's label enum.
    edge_type = create_model(
        f"ConfiguredGatheredEvidenceEdgeAgenticV1_{label_suffix}",
        __base__=GatheredEvidenceEdgeAgenticV1,
        relationship_type=(
            allowed_labels,
            Field(description="configured relationship between this evidence and the target goal's auditable claim"),
        ),
    )
    return cast(
        type[GatherEvidenceForAgenticGraphActionV1],
        # Preserve the base action and its validators while making evidence_edges use the configured edge schema.
        create_model(
            f"ConfiguredGatherEvidenceForAgenticGraphActionV1_{label_suffix}",
            __base__=GatherEvidenceForAgenticGraphActionV1,
            evidence_edges=(
                list[edge_type],  # type: ignore[valid-type]  # Runtime Pydantic model returned by create_model.
                Field(description="evidence relationships using only the configured relationship labels"),
            ),
        ),
    )


class AgenticGraphObservation(Observation):
    graph: str

    @property
    def to_llm_content(self) -> list[TextContent]:
        return [TextContent(text=self.graph)]


class AgenticGraphConversation(LocalConversation):
    """Ephemeral OpenHands conversation carrying its in-progress confidence graph."""

    confidence_graph: ConfidenceGraph


def apply_agentic_graph_action(graph: ConfidenceGraph, action: AgenticGraphAction) -> ConfidenceGraph:
    """Apply one strategy action atomically to an in-progress confidence graph."""
    target_goals = [
        node
        for node in graph.nodes
        if isinstance(node, GSNGoalNode) and node.goal_name == action.target_goal_identifier
    ]
    if len(target_goals) != 1:
        raise ValueError(
            f"Expected exactly one target goal named {action.target_goal_identifier!r}, found {len(target_goals)}"
        )
    target_goal = target_goals[0]

    nodes_by_id = {node.id: node for node in graph.nodes}
    if any(edge.target == target_goal.id and isinstance(nodes_by_id[edge.source], GSNGoalNode) for edge in graph.edges):
        raise ValueError(f"Goal {action.target_goal_identifier!r} has more than one strategy")

    if isinstance(action, DivideAndConquerAction):
        child_specs = action.sub_goals
        relationship_type: GoalEdgeLabel = "decomposes_from"
    else:
        child_specs = [action.particularized_goal]
        relationship_type = "particularizes"

    existing_goal_names = {node.goal_name for node in graph.nodes if isinstance(node, GSNGoalNode)}
    child_goal_names = [child.goal_identifier for child in child_specs]
    duplicate_child_goal_names = sorted({name for name in child_goal_names if child_goal_names.count(name) > 1})
    if duplicate_child_goal_names:
        raise ValueError(f"Duplicate child goal names: {duplicate_child_goal_names}")
    reused_goal_names = sorted(existing_goal_names.intersection(child_goal_names))
    if reused_goal_names:
        raise ValueError(f"Duplicate goal names: {reused_goal_names}")

    child_goals = [
        GSNGoalNode(
            goal_name=child.goal_identifier,
            auditable_claim=child.auditable_claim,
            reasoning=child.reasoning,
        )
        for child in child_specs
    ]
    edges = [
        ConfidenceEdge(source=child.id, target=target_goal.id, relationship_type=relationship_type)
        for child in child_goals
    ]
    existing_nodes = graph.nodes
    if isinstance(action, (DivideAndConquerInterpAction, ParticularizeInterpAction)):
        target_goal = target_goal.model_copy(update={"confidence_in_children": action.confidence})
        existing_nodes = [target_goal if node.id == target_goal.id else node for node in graph.nodes]
    return ConfidenceGraph(
        nodes=[*existing_nodes, *child_goals],
        edges=[*graph.edges, *edges],
        goal_zero_node_id=graph.goal_zero_node_id,
    )


def apply_agentic_evidence_action(
    graph: ConfidenceGraph,
    action: GatherEvidenceForAgenticGraphActionV1,
) -> ConfidenceGraph:
    """Add V1 LiteLLM-V3-shaped evidence to the graph's current leaf goals."""
    leaf_goals = [node for node in get_leaves(graph) if isinstance(node, GSNGoalNode)]
    goals_by_identifier = {goal.goal_name: goal for goal in leaf_goals}
    if len(goals_by_identifier) != len(leaf_goals):
        raise ValueError("Duplicate leaf goal identifiers")

    target_goal_identifiers = {edge.target_goal_identifier for edge in action.evidence_edges}
    unknown_goal_identifiers = sorted(target_goal_identifiers - set(goals_by_identifier))
    if unknown_goal_identifiers:
        raise ValueError(f"Unknown target goal identifiers: {unknown_goal_identifiers}")

    used_evidence_keys = {edge.evidence_key for edge in action.evidence_edges}
    evidence_nodes_by_key = {
        item.evidence_key: EvidenceNodeV2(
            evidence=item.evidence,
            step_numbers=item.step_numbers,
            auditable_claim=item.auditable_claim,
            contribution=item.contribution,
        )
        for item in action.evidence_catalog
        if item.evidence_key in used_evidence_keys
    }
    evidence_edges = [
        ConfidenceEdge(
            source=evidence_nodes_by_key[edge.evidence_key].id,
            target=goals_by_identifier[edge.target_goal_identifier].id,
            relationship_type=edge.relationship_type,
        )
        for edge in action.evidence_edges
    ]
    return ConfidenceGraph(
        nodes=[*graph.nodes, *evidence_nodes_by_key.values()],
        edges=[*graph.edges, *evidence_edges],
        goal_zero_node_id=graph.goal_zero_node_id,
    )


class _AgenticGraphExecutor(ToolExecutor[AgenticGraphAction, AgenticGraphObservation]):
    def __call__(self, action: AgenticGraphAction, conversation=None) -> AgenticGraphObservation:
        if not isinstance(conversation, AgenticGraphConversation):
            raise ValueError("Agentic graph tools require an AgenticGraphConversation")
        conversation.confidence_graph = apply_agentic_graph_action(conversation.confidence_graph, action)
        rendered_graph = render_graph_claims_compact(conversation.confidence_graph)
        logger.info(
            "Agentic graph tool result for %s targeting %r:\n%s",
            type(action).__name__,
            action.target_goal_identifier,
            rendered_graph,
        )
        return AgenticGraphObservation(graph=rendered_graph)


class _AgenticEvidenceExecutor(ToolExecutor[GatherEvidenceForAgenticGraphActionV1, AgenticGraphObservation]):
    def __call__(self, action: GatherEvidenceForAgenticGraphActionV1, conversation=None) -> AgenticGraphObservation:
        if not isinstance(conversation, AgenticGraphConversation):
            raise ValueError("Agentic graph tools require an AgenticGraphConversation")
        conversation.confidence_graph = apply_agentic_evidence_action(conversation.confidence_graph, action)
        rendered_graph = render_graph_claims_compact(conversation.confidence_graph)
        logger.info("Agentic evidence tool result:\n%s", rendered_graph)
        return AgenticGraphObservation(graph=rendered_graph)


class DivideAndConquerTool(ToolDefinition):
    name = "divide_and_conquer"

    @classmethod
    def create(cls, **kwargs) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Decompose one existing claim into necessary-and-jointly-sufficient subclaims at approximately "
                    "the same abstraction level."
                ),
                action_type=DivideAndConquerAction,
                observation_type=AgenticGraphObservation,
                executor=_AgenticGraphExecutor(),
            )
        ]


class ParticularizeTool(ToolDefinition):
    name = "particularize"

    @classmethod
    def create(cls, **kwargs) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Restate one abstract, task-agnostic claim as an equivalent concrete, task-aware claim without "
                    "changing its truth condition."
                ),
                action_type=ParticularizeAction,
                observation_type=AgenticGraphObservation,
                executor=_AgenticGraphExecutor(),
            )
        ]


class DivideAndConquerInterpTool(ToolDefinition):
    name = "divide_and_conquer_interp"

    @classmethod
    def create(cls, **kwargs) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Decompose one existing claim into necessary-and-jointly-sufficient subclaims at approximately "
                    "the same abstraction level. Report confidence in the decomposition: that likelihood that all "
                    "children are in fact collectively sufficient, conditionally independent, and necessary."
                ),
                action_type=DivideAndConquerInterpAction,
                observation_type=AgenticGraphObservation,
                executor=_AgenticGraphExecutor(),
            )
        ]


class ParticularizeInterpTool(ToolDefinition):
    name = "particularize_interp"

    @classmethod
    def create(cls, **kwargs) -> Sequence[Self]:
        return [
            cls(
                description=(
                    "Restate one abstract, task-agnostic claim as an equivalent concrete, task-aware claim without "
                    "changing its truth condition. Report confidence in the particularization: the likelihood that "
                    "the restatement is exactly equivalent in meaning in this agent's task context."
                ),
                action_type=ParticularizeInterpAction,
                observation_type=AgenticGraphObservation,
                executor=_AgenticGraphExecutor(),
            )
        ]


class GatherEvidenceForAgenticGraphToolV1(ToolDefinition):
    """Versioned evidence tool for the live agentic graph, distinct from replay-oriented OH tools."""

    name = "gather_evidence_for_agentic_graph_v1"

    @classmethod
    def create(
        cls,
        *,
        evidence_edge_labels: Sequence[EvidenceEdgeLabel] | None = None,
        **kwargs,
    ) -> Sequence[Self]:
        if evidence_edge_labels is None:
            evidence_edge_labels = ["proves", "supports", "refutes", "undermines"]
        return [
            cls(
                description=(
                    "Strategy: gather reusable evidence from the completed trajectory and connect it to all "
                    "target sub-goals with typed evidence edges"
                ),
                action_type=configured_gather_evidence_action_type(evidence_edge_labels),
                observation_type=AgenticGraphObservation,
                executor=_AgenticEvidenceExecutor(),
            )
        ]


register_tool(DivideAndConquerTool.name, DivideAndConquerTool)
register_tool(ParticularizeTool.name, ParticularizeTool)
register_tool(DivideAndConquerInterpTool.name, DivideAndConquerInterpTool)
register_tool(ParticularizeInterpTool.name, ParticularizeInterpTool)
register_tool(GatherEvidenceForAgenticGraphToolV1.name, GatherEvidenceForAgenticGraphToolV1)
