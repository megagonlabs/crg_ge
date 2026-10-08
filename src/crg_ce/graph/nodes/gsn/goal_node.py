from pydantic import Field, field_validator

from crg_ce.graph.nodes.base_node import CENode


class GSNGoalNode(CENode):
    goal_name: str = Field(description="name describing the achievable goal")
    auditable_claim: str = Field(
        description="the goal phrased as an auditable claim, which is true if the goal is achieved and false otherwise"
    )
    reasoning: str = Field(description="justification for the inclusion of this goal")
    confidence_in_children: float = Field(
        default=-1,
        description="Confidence that this goal's children correctly relate to it: -1 or between [0, 1]",
    )

    @field_validator("confidence_in_children")
    @classmethod
    def validate_confidence_in_children(cls, value: float) -> float:
        if value == -1 or 0 <= value <= 1:
            return value
        raise ValueError("confidence_in_children must be -1 or between 0 and 1")


class EvidenceNodeV2(CENode):
    evidence: str = Field(description="brief summary of this piece of evidence")
    step_numbers: list[int] = Field(description="the step numbers which make up this piece of evidence")
    auditable_claim: str = Field(description="a restatement of this evidence as a falsifiable claim")
    contribution: str = Field(description="reasoning describing how this evidence contributes to the confidence graph")
