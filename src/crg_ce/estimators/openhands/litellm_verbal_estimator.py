"""Estimate confidence ex situ from a rendered OpenHands trajectory using LiteLLM."""

import asyncio
import math
import re
from collections.abc import Callable
from logging import Logger
from pathlib import Path

import litellm
from jinja2 import Template
from litellm.types.utils import TopLogprob
from openhands.sdk import get_logger
from pydantic import BaseModel
from tenacity import AsyncRetrying, RetryError, Retrying, retry_if_exception_type, stop_after_attempt

from crg_ce.estimators.base_estimator import (
    BaseConfidenceEstimator,
    ConfEstimationInput,
    ConfEstimationOutput,
    model_usage_from_stats,
    rescale_confidence,
)
from crg_ce.estimators.openhands.config import LiteLLMVerbalEstimatorConfig
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.resources import read_resource
from crg_ce.utils.general import resolve_template
from crg_ce.utils.litellm_utils import (
    LiteLLMCallStats,
    acomplete_structured,
    acomplete_text,
    complete_structured,
    complete_text,
    resolve_api_key,
)
from crg_ce.utils.openhands import load_conversation_state_and_events_from_archive
from crg_ce.utils.openhands_trajectory import render_trajectory
from crg_ce.utils.prompt_logging import log_prompt

SAMPLED_TRUE_OR_FALSE_OUTPUT_INSTRUCTION = (
    "Reason about the correctness of the completed work, then end your response with your answer on its own line "
    'in exactly this form: "Successful: <answer>", where <answer> is either True or False.'
)


class StructuredConfidence(BaseModel):
    confidence: float
    rationale: str


def parse_confidence_percentage(response: str) -> float:
    match = re.search(r"(?:^|\n)Confidence:\s*(\d+)%\s*\Z", response)
    if match is None:
        raise ValueError("Response does not end with a whole-number 'Confidence: N%'")
    percentage = float(match.group(1))
    if not 0 <= percentage <= 100:
        raise ValueError(f"Confidence percentage must be between 0 and 100: {percentage}")
    return percentage / 100


def confidence_from_true_or_false_samples(samples: list[str], expected_samples: int) -> float:
    if len(samples) != expected_samples:
        raise ValueError(f"Expected {expected_samples} True/False samples, received {len(samples)}")
    answers: list[str] = []
    malformed_samples = 0
    for sample in samples:
        match = re.search(r"(?:^|\n)Successful:\s*(True|False)\s*\Z", sample)
        if match is None:
            malformed_samples += 1
        else:
            answers.append(match.group(1))
    if malformed_samples >= 3:
        raise ValueError(f"Received {malformed_samples} malformed True/False samples")
    return answers.count("True") / len(answers)


def confidence_from_top_logprobs(top_logprobs: list[TopLogprob]) -> float:
    logprobs_by_label: dict[str, list[float]] = {"True": [], "False": []}
    seen_raw_tokens: set[str] = set()
    for token_logprob in top_logprobs:
        label = token_logprob.token.strip()
        if label not in logprobs_by_label:
            continue
        if token_logprob.token in seen_raw_tokens:
            raise ValueError(f"Duplicate raw top log-probability token: {token_logprob.token!r}")
        seen_raw_tokens.add(token_logprob.token)
        logprobs_by_label[label].append(token_logprob.logprob)

    missing_tokens = {label for label, logprobs in logprobs_by_label.items() if not logprobs}
    if missing_tokens:
        raise ValueError(f"Top log-probability tokens are missing {sorted(missing_tokens)}")

    max_logprob = max(logprob for logprobs in logprobs_by_label.values() for logprob in logprobs)
    true_probability = sum(math.exp(logprob - max_logprob) for logprob in logprobs_by_label["True"])
    false_probability = sum(math.exp(logprob - max_logprob) for logprob in logprobs_by_label["False"])
    return true_probability / (true_probability + false_probability)


class LiteLLMVerbalEstimator(BaseConfidenceEstimator):
    """Estimate confidence with a fresh LLM call over a completed agent trajectory."""

    cfg: LiteLLMVerbalEstimatorConfig
    logger: Logger
    instruction_template: Template
    true_or_false_template: Template
    ask_and_parse_output_instruction: str

    def __init__(self, cfg: LiteLLMVerbalEstimatorConfig, *, llm_limiter: LLMCallLimiter | None = None) -> None:
        self.cfg = cfg
        self.logger = get_logger(__name__)
        self.instruction_template = resolve_template(cfg.instruction_template)
        self.true_or_false_template = resolve_template("prompts/confidence_estimation/litellm/true_or_false.j2")
        self.ask_and_parse_output_instruction = read_resource(cfg.ask_and_parse_output_instruction).strip()
        self.llm_limiter = llm_limiter

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        state, events = load_conversation_state_and_events_from_archive(ce_input.conversation_archive_path)
        trajectory = render_trajectory(events, state=state, start_at_first_action_event=True)
        stats = LiteLLMCallStats()
        domain_success_criteria = self.cfg.render_domain_success_criteria(ce_input.benchmark)

        if self.cfg.query_type == "structured":
            complete_and_validate = self._prepare_structured_completion(
                trajectory, ce_input.problem_statement, domain_success_criteria, stats, ce_input.output_dir
            )
        elif self.cfg.query_type == "ask_and_parse":
            complete_and_validate = self._prepare_ask_and_parse_completion(
                trajectory, ce_input.problem_statement, domain_success_criteria, stats, ce_input.output_dir
            )
        elif self.cfg.query_type == "true_or_false":
            complete_and_validate = self._prepare_true_or_false_completion(
                trajectory, ce_input.problem_statement, domain_success_criteria, stats, ce_input.output_dir
            )
        elif self.cfg.query_type == "sampled_true_or_false":
            complete_and_validate = self._prepare_sampled_true_or_false_completion(
                trajectory, ce_input.problem_statement, domain_success_criteria, stats, ce_input.output_dir
            )
        else:
            raise ValueError(f"Unsupported query_type={self.cfg.query_type}")

        output = self.robust_complete_and_validate(complete_and_validate, stats)
        self.logger.info("Scored an ex-situ verbalized confidence of %s", output.confidence)
        self.save_output(output, ce_input.output_dir)
        return output

    async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        if self.llm_limiter is None:
            return await super().aestimate_confidence(ce_input)
        llm_limiter = self.llm_limiter
        state, events = await asyncio.to_thread(
            load_conversation_state_and_events_from_archive, ce_input.conversation_archive_path
        )
        trajectory = render_trajectory(events, state=state, start_at_first_action_event=True)
        stats = LiteLLMCallStats()
        domain_success_criteria = self.cfg.render_domain_success_criteria(ce_input.benchmark)
        scale_min = self.cfg.verbalization.scale_min
        scale_max = self.cfg.verbalization.scale_max
        if self.cfg.query_type == "structured":
            instruction = self._render_instruction(
                trajectory,
                problem_statement=ce_input.problem_statement,
                domain_success_criteria=domain_success_criteria,
                output_instruction=(
                    f"Return a structured confidence score from {scale_min} to {scale_max}, where "
                    f"{scale_min} means no confidence and {scale_max} means complete confidence, along with "
                    "a concise rationale."
                ),
            )
        elif self.cfg.query_type == "ask_and_parse":
            instruction = self._render_instruction(
                trajectory,
                problem_statement=ce_input.problem_statement,
                domain_success_criteria=domain_success_criteria,
                output_instruction=self.ask_and_parse_output_instruction,
            )
        elif self.cfg.query_type == "true_or_false":
            instruction = self.true_or_false_template.render(
                trajectory=trajectory,
                problem_statement=ce_input.problem_statement,
                domain_success_criteria=domain_success_criteria,
            )
        elif self.cfg.query_type == "sampled_true_or_false":
            instruction = self._render_instruction(
                trajectory,
                problem_statement=ce_input.problem_statement,
                domain_success_criteria=domain_success_criteria,
                output_instruction=SAMPLED_TRUE_OR_FALSE_OUTPUT_INSTRUCTION,
            )
        else:
            raise ValueError(f"Unsupported query_type={self.cfg.query_type}")
        log_prompt(instruction, ce_input.output_dir)

        async def complete_and_validate() -> float:
            if self.cfg.query_type == "structured":
                result = await acomplete_structured(
                    model=self.cfg.agent.model_name,
                    messages=[{"role": "user", "content": instruction}],
                    output_model=StructuredConfidence,
                    llm_limiter=llm_limiter,
                    api_key=self.cfg.agent.api_key,
                    base_url=self.cfg.agent.api_base,
                    top_p=self.cfg.agent.top_p,
                    reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore[arg-type]
                    allowed_openai_params=self.cfg.agent.allowed_openai_params,
                    max_completion_tokens=self.cfg.agent.max_output_tokens,
                    timeout=self.cfg.agent.timeout,
                    stats=stats,
                )
                return rescale_confidence(min_score=scale_min, max_score=scale_max, score=result.confidence)
            if self.cfg.query_type == "ask_and_parse":
                response = await acomplete_text(
                    model=self.cfg.agent.model_name,
                    messages=[{"role": "user", "content": instruction}],
                    llm_limiter=llm_limiter,
                    api_key=self.cfg.agent.api_key,
                    base_url=self.cfg.agent.api_base,
                    top_p=self.cfg.agent.top_p,
                    reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore[arg-type]
                    allowed_openai_params=self.cfg.agent.allowed_openai_params,
                    max_completion_tokens=self.cfg.agent.max_output_tokens,
                    timeout=self.cfg.agent.timeout,
                    stats=stats,
                )
                return parse_confidence_percentage(response)
            if self.cfg.query_type == "true_or_false":
                async with llm_limiter.slot():
                    response = await litellm.acompletion(
                        model=self.cfg.agent.model_name,
                        messages=[
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": "Answer:"},
                        ],
                        api_key=resolve_api_key(self.cfg.agent.api_key),
                        base_url=self.cfg.agent.api_base,
                        temperature=0,
                        top_p=self.cfg.agent.top_p,
                        reasoning_effort=self.cfg.agent.reasoning_effort,
                        allowed_openai_params=self.cfg.agent.allowed_openai_params,
                        max_completion_tokens=1,
                        timeout=self.cfg.agent.timeout,
                        logprobs=True,
                        top_logprobs=10,
                        extra_body={
                            "continue_final_message": True,
                            "add_generation_prompt": False,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                stats.record_response(response, model=self.cfg.agent.model_name)
                choice_logprobs = response.choices[0].logprobs  # pyright: ignore[reportAttributeAccessIssue]
                if choice_logprobs is None or not choice_logprobs.content:
                    raise ValueError("LiteLLM response is missing token log probabilities")
                return confidence_from_top_logprobs(choice_logprobs.content[0].top_logprobs)
            if self.cfg.query_type == "sampled_true_or_false":
                async with llm_limiter.slot():
                    response = await litellm.acompletion(
                        model=self.cfg.agent.model_name,
                        messages=[{"role": "user", "content": instruction}],
                        api_key=resolve_api_key(self.cfg.agent.api_key),
                        base_url=self.cfg.agent.api_base,
                        top_p=self.cfg.agent.top_p,
                        reasoning_effort=self.cfg.agent.reasoning_effort,
                        allowed_openai_params=self.cfg.agent.allowed_openai_params,
                        max_completion_tokens=self.cfg.agent.max_output_tokens,
                        timeout=self.cfg.agent.timeout,
                        n=self.cfg.true_or_false_samples,
                        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                        **self.cfg.agent.completion_kwargs,
                    )
                stats.record_response(response, model=self.cfg.agent.model_name)
                samples: list[str] = []
                for choice in response.choices:
                    content = choice.message.content
                    if not isinstance(content, str):
                        raise ValueError("LiteLLM response is missing text content")
                    samples.append(content)
                return confidence_from_true_or_false_samples(samples, self.cfg.true_or_false_samples)
            raise ValueError(f"Unsupported query_type={self.cfg.query_type}")

        try:
            async for attempt in AsyncRetrying(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError)):
                with attempt:
                    confidence = await complete_and_validate()
        except RetryError as exc:
            raise ValueError("Could not obtain valid confidence after 3 attempts") from exc

        output = ConfEstimationOutput(
            confidence=confidence,
            total_tokens=stats.total_tokens,
            generated_tokens=stats.completion_tokens,
            cost=stats.cost,
            usage_by_model=model_usage_from_stats(stats),
        )
        self.save_output(output, ce_input.output_dir)
        return output

    def robust_complete_and_validate(
        self,
        complete_and_validate: Callable[[], float],
        stats: LiteLLMCallStats,
    ) -> ConfEstimationOutput:
        retryer = Retrying(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError))
        try:
            confidence = retryer(complete_and_validate)
        except RetryError as exc:
            raise ValueError("Could not obtain valid confidence after 3 attempts") from exc

        return ConfEstimationOutput(
            confidence=confidence,
            total_tokens=stats.total_tokens,
            generated_tokens=stats.completion_tokens,
            cost=stats.cost,
            usage_by_model=model_usage_from_stats(stats),
        )

    def _prepare_structured_completion(
        self,
        trajectory: str,
        problem_statement: str,
        domain_success_criteria: str | None,
        stats: LiteLLMCallStats,
        output_dir: Path,
    ) -> Callable[[], float]:
        scale_min = self.cfg.verbalization.scale_min
        scale_max = self.cfg.verbalization.scale_max
        instruction = self._render_instruction(
            trajectory,
            problem_statement=problem_statement,
            domain_success_criteria=domain_success_criteria,
            output_instruction=(
                f"Return a structured confidence score from {scale_min} to {scale_max}, where {scale_min} means "
                f"no confidence and {scale_max} means complete confidence, along with a concise rationale."
            ),
        )
        log_prompt(instruction, output_dir)

        def complete_and_validate() -> float:
            result = complete_structured(
                model=self.cfg.agent.model_name,
                messages=[{"role": "user", "content": instruction}],
                output_model=StructuredConfidence,
                api_key=self.cfg.agent.api_key,
                base_url=self.cfg.agent.api_base,
                top_p=self.cfg.agent.top_p,
                reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore
                allowed_openai_params=self.cfg.agent.allowed_openai_params,
                max_completion_tokens=self.cfg.agent.max_output_tokens,
                timeout=self.cfg.agent.timeout,
                stats=stats,
            )
            if not scale_min <= result.confidence <= scale_max:
                raise ValueError(f"Confidence must be between {scale_min} and {scale_max}: {result.confidence}")
            return rescale_confidence(min_score=scale_min, max_score=scale_max, score=result.confidence)

        return complete_and_validate

    def _prepare_ask_and_parse_completion(
        self,
        trajectory: str,
        problem_statement: str,
        domain_success_criteria: str | None,
        stats: LiteLLMCallStats,
        output_dir: Path,
    ) -> Callable[[], float]:
        instruction = self._render_instruction(
            trajectory,
            problem_statement=problem_statement,
            domain_success_criteria=domain_success_criteria,
            output_instruction=self.ask_and_parse_output_instruction,
        )
        log_prompt(instruction, output_dir)

        def complete_and_parse() -> float:
            response = complete_text(
                model=self.cfg.agent.model_name,
                messages=[{"role": "user", "content": instruction}],
                api_key=self.cfg.agent.api_key,
                base_url=self.cfg.agent.api_base,
                top_p=self.cfg.agent.top_p,
                reasoning_effort=self.cfg.agent.reasoning_effort,  # type: ignore
                allowed_openai_params=self.cfg.agent.allowed_openai_params,
                max_completion_tokens=self.cfg.agent.max_output_tokens,
                timeout=self.cfg.agent.timeout,
                stats=stats,
            )
            return parse_confidence_percentage(response)

        return complete_and_parse

    def _prepare_true_or_false_completion(
        self,
        trajectory: str,
        problem_statement: str,
        domain_success_criteria: str | None,
        stats: LiteLLMCallStats,
        output_dir: Path,
    ) -> Callable[[], float]:
        instruction = self.true_or_false_template.render(
            trajectory=trajectory,
            problem_statement=problem_statement,
            domain_success_criteria=domain_success_criteria,
        )
        log_prompt(instruction, output_dir)

        def complete_and_score() -> float:
            response = litellm.completion(
                model=self.cfg.agent.model_name,
                messages=[
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": "Answer:"},
                ],
                api_key=resolve_api_key(self.cfg.agent.api_key),
                base_url=self.cfg.agent.api_base,
                temperature=0,
                top_p=self.cfg.agent.top_p,
                reasoning_effort=self.cfg.agent.reasoning_effort,  # pyright: ignore[reportArgumentType]
                allowed_openai_params=self.cfg.agent.allowed_openai_params,
                max_completion_tokens=1,
                timeout=self.cfg.agent.timeout,
                logprobs=True,
                top_logprobs=10,
                extra_body={
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            stats.record_response(response, model=self.cfg.agent.model_name)

            choice_logprobs = response.choices[0].logprobs  # pyright: ignore[reportAttributeAccessIssue]
            if choice_logprobs is None or not choice_logprobs.content:
                raise ValueError("LiteLLM response is missing token log probabilities")
            return confidence_from_top_logprobs(choice_logprobs.content[0].top_logprobs)

        return complete_and_score

    def _prepare_sampled_true_or_false_completion(
        self,
        trajectory: str,
        problem_statement: str,
        domain_success_criteria: str | None,
        stats: LiteLLMCallStats,
        output_dir: Path,
    ) -> Callable[[], float]:
        instruction = self._render_instruction(
            trajectory,
            problem_statement=problem_statement,
            domain_success_criteria=domain_success_criteria,
            output_instruction=SAMPLED_TRUE_OR_FALSE_OUTPUT_INSTRUCTION,
        )
        log_prompt(instruction, output_dir)

        def complete_and_score() -> float:
            response = litellm.completion(
                model=self.cfg.agent.model_name,
                messages=[{"role": "user", "content": instruction}],
                api_key=resolve_api_key(self.cfg.agent.api_key),
                base_url=self.cfg.agent.api_base,
                top_p=self.cfg.agent.top_p,
                reasoning_effort=self.cfg.agent.reasoning_effort,  # pyright: ignore[reportArgumentType]
                allowed_openai_params=self.cfg.agent.allowed_openai_params,
                max_completion_tokens=self.cfg.agent.max_output_tokens,
                timeout=self.cfg.agent.timeout,
                n=self.cfg.true_or_false_samples,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
                **self.cfg.agent.completion_kwargs,
            )
            stats.record_response(response, model=self.cfg.agent.model_name)
            samples: list[str] = []
            for choice in response.choices:
                content = choice.message.content
                if not isinstance(content, str):
                    raise ValueError("LiteLLM response is missing text content")
                samples.append(content)
            return confidence_from_true_or_false_samples(samples, self.cfg.true_or_false_samples)

        return complete_and_score

    def _render_instruction(
        self,
        trajectory: str,
        *,
        problem_statement: str,
        domain_success_criteria: str | None,
        output_instruction: str,
    ) -> str:
        return self.instruction_template.render(
            trajectory=trajectory,
            problem_statement=problem_statement,
            domain_success_criteria=domain_success_criteria,
            output_instruction=output_instruction,
            scale_min=self.cfg.verbalization.scale_min,
            scale_max=self.cfg.verbalization.scale_max,
        )
