from pydantic import BaseModel, Field


class ConfidenceEstimateLiteLLM(BaseModel):
    """Structured confidence response used by the LiteLLM graph populator."""

    confidence: float = Field(description="calibrated confidence on the scale specified in the prompt")
    rationale: str = Field(description="brief justification for the confidence estimate")
