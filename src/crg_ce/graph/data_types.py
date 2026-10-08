from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crg_ce.graph.edges import ConfidenceEdge
from crg_ce.graph.nodes import CENode  # noqa: F403


class ConfidenceGraph(BaseModel):
    """
    Our core (serializable) graph representation for confidence estimation
    """

    nodes: list[CENode]
    edges: list[ConfidenceEdge]
    goal_zero_node_id: str | None = Field(default=None, exclude_if=lambda value: value is None)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def validate_edge_node_references(self) -> "ConfidenceGraph":
        node_ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"Unknown edge source id: {edge.source}")
            if edge.target not in node_ids:
                raise ValueError(f"Unknown edge target id: {edge.target}")
        if self.goal_zero_node_id is not None and self.goal_zero_node_id not in node_ids:
            raise ValueError(f"Unknown goal_zero_node_id: {self.goal_zero_node_id}")
        return self
