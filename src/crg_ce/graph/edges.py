from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

EvidenceEdgeLabel = Literal["proves", "supports", "refutes", "undermines", "unverified"]
GoalEdgeLabel = Literal["decomposes_from", "particularizes"]

_EVIDENCE_EDGE_LABEL_DESCRIPTIONS: dict[EvidenceEdgeLabel, str] = {
    "proves": "establishes the target claim as true beyond reasonable doubt",
    "supports": "increases confidence in the target claim",
    "refutes": "establishes the target claim as false beyond reasonable doubt",
    "undermines": "decreases confidence in the target claim",
    "unverified": "identifies missing verification or evidence, without indicating falsehood",
}


def get_edge_label_descriptions(
    edge_labels: list[EvidenceEdgeLabel],
) -> dict[EvidenceEdgeLabel, str]:
    return {edge_label: _EVIDENCE_EDGE_LABEL_DESCRIPTIONS[edge_label] for edge_label in edge_labels}


GOAL_EDGE_LABEL_DESCRIPTIONS: dict[GoalEdgeLabel, str] = {
    "decomposes_from": "is a conjunct in a decomposition of the target goal",
    "particularizes": "restates the target goal concretely for the specific task context",
}


EDGE_EDGE_RELATIONSHIPS: dict[str, str] = {}


class ConfidenceEdge(BaseModel):
    source: str
    target: str
    relationship_type: EvidenceEdgeLabel | GoalEdgeLabel | None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
