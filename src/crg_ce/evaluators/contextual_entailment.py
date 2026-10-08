from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict

from crg_ce.estimators.openhands.config import AgentConfig, EntailmenntPromptConfig
from crg_ce.resources import read_resource
from crg_ce.utils.litellm_utils import LiteLLMCallStats, tenaciously_complete_structured

EntailmentLabel = Literal["entailment", "contradiction", "neutral"]


class EntailmentAssessment(BaseModel):
    label: EntailmentLabel
    rationale: str

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ContextualEntailmentInputs(BaseModel):
    premise: str
    hypothesis: str
    examples: str | None = None
    agent_task_input: str | None = None
    trajectory: str | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class ContextualEntailmentEvaluator:
    """Classify whether a premise entails, contradicts, or is neutral toward a hypothesis."""

    examples: str | None

    def __init__(
        self,
        agent: AgentConfig,
        prompts: EntailmenntPromptConfig,
    ) -> None:
        self.agent = agent
        self.system_prompt = prompts.system_prompt
        self.task_prompt = prompts.entailment_task_prompt
        self.examples = read_resource(prompts.examples.as_posix()) if prompts.examples is not None else None

    def evaluate(
        self,
        premise: str,
        hypothesis: str,
        *,
        examples: str | None = None,
        agent_task_input: str | None = None,
        trajectory: str | None = None,
        stats: LiteLLMCallStats | None = None,
    ) -> EntailmentAssessment:
        inputs = ContextualEntailmentInputs(
            premise=premise,
            hypothesis=hypothesis,
            examples=examples if examples is not None else self.examples,
            agent_task_input=agent_task_input,
            trajectory=trajectory,
        )
        return tenaciously_complete_structured(
            model=self.agent.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt.render()},
                {"role": "user", "content": self.task_prompt.render(**inputs.model_dump())},
            ],
            output_model=EntailmentAssessment,
            api_key=self.agent.api_key,
            base_url=self.agent.api_base,
            top_p=self.agent.top_p,
            reasoning_effort=self.agent.reasoning_effort,  # type: ignore[arg-type]
            max_completion_tokens=self.agent.max_output_tokens,
            stats=stats,
        )
