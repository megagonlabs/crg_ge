import uuid
from abc import ABC
from typing import ClassVar

from openhands.sdk.utils.models import DiscriminatedUnionMixin
from pydantic import ConfigDict, Field, field_validator


class CENode(DiscriminatedUnionMixin, ABC):
    """
    Base Node type for our graphs
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique node id (ULID/UUID)",
    )
    confidence: float = Field(
        default=-1,
        description="Confidence value: -1 or between [0, 1]",
    )
    confidence_rationale: str | None = Field(
        default=None,
        description="Brief rationale produced when assigning this node's confidence",
    )

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float) -> float:
        if value == -1 or 0 <= value <= 1:
            return value
        raise ValueError("confidence must be -1 or between 0 and 1")

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
