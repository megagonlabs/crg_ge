"""Estimate success from the likelihood of an OpenHands trajectory."""

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import litellm
from openai import AsyncOpenAI, OpenAI
from openhands.sdk import get_logger
from openhands.sdk.event import SystemPromptEvent
from pydantic import BaseModel
from scipy.special import logsumexp
from tenacity import AsyncRetrying, Retrying, retry_if_exception_type, stop_after_attempt

from crg_ce.estimators.base_estimator import (
    BaseConfidenceEstimator,
    ConfEstimationInput,
    ConfEstimationOutput,
    model_usage_from_stats,
)
from crg_ce.estimators.openhands.config import LogProbsEstimatorConfig
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.utils.litellm_utils import LiteLLMCallStats, resolve_api_key
from crg_ce.utils.openhands import build_llm, load_conversation_state_and_events_from_archive
from crg_ce.utils.openhands_trajectory import get_active_branch_events, trajectory_events_to_messages
from crg_ce.utils.prompt_logging import log_prompt


class LogProbFeatures(BaseModel):
    """Token-aligned features retained for later sliced-sequence predictors."""

    token_ids: list[int]
    tokens: list[str | None]
    token_log_probs: list[float | None]
    mean_log_prob: float


def _prompt_token_features(response: Any) -> tuple[list[int], list[str | None], list[float | None]]:
    token_ids = response.prompt_token_ids
    prompt_logprobs = response.prompt_logprobs
    if not isinstance(token_ids, list) or not all(isinstance(token_id, int) for token_id in token_ids):
        raise TypeError("The response does not contain prompt token IDs")
    if not isinstance(prompt_logprobs, list):
        raise TypeError("The response does not contain prompt log-probabilities")
    if len(prompt_logprobs) != len(token_ids):
        raise ValueError("The log-probability response token IDs and scores are not aligned")

    tokens: list[str | None] = []
    chosen_log_probs: list[float | None] = []
    for index, (token_id, alternatives) in enumerate(zip(token_ids, prompt_logprobs, strict=True)):
        if alternatives is None:
            if index != 0:
                raise ValueError(f"Missing prompt log-probability at token index {index}")
            tokens.append(None)
            chosen_log_probs.append(None)
            continue

        token_data = alternatives.get(token_id, alternatives.get(str(token_id)))
        if token_data is None:
            raise ValueError(f"Prompt log-probabilities omit chosen token {token_id} at index {index}")
        log_prob = token_data.get("logprob") if isinstance(token_data, dict) else token_data.logprob
        if log_prob is None:
            raise ValueError(f"Chosen token {token_id} has no log-probability at index {index}")
        decoded_token = token_data.get("decoded_token") if isinstance(token_data, dict) else token_data.decoded_token
        if not isinstance(decoded_token, str):
            raise ValueError(f"Chosen token {token_id} has no decoded token at index {index}")
        tokens.append(decoded_token)
        chosen_log_probs.append(float(log_prob))
    return token_ids, tokens, chosen_log_probs


def _chatml_action_regions(rendered_prompt: str) -> list[list[tuple[int, int]]]:
    """Recover one contiguous range per assistant message from ChatML's `<|im_start|>assistant` blocks."""
    action_start_marker = "<|im_start|>assistant"
    action_end_marker = "<|im_end|>"
    action_starts: list[int] = []
    search_start = 0
    # first: in the prompt in combined string form, find the character indices for the start and end of each action span
    while (action_start := rendered_prompt.find(action_start_marker, search_start)) != -1:
        action_starts.append(action_start)
        search_start = action_start + len(action_start_marker)

    action_regions: list[list[tuple[int, int]]] = []
    unclosed_action_starts: list[int] = []
    for action_start in action_starts:
        action_end = rendered_prompt.find(action_end_marker, action_start + len(action_start_marker))
        if action_end == -1:
            unclosed_action_starts.append(action_start)
        else:
            action_regions.append([(action_start, action_end + len(action_end_marker))])

    if action_starts and unclosed_action_starts not in ([], [action_starts[-1]]):
        raise ValueError(f"Unexpected unclosed assistant action spans at {unclosed_action_starts}")
    return action_regions


def _action_regions(rendered_prompt: str, model_name: str) -> list[list[tuple[int, int]]]:
    """The character ranges each assistant message contributed to the prompt the surrogate model rendered."""
    surrogate = model_name.lower()
    if "qwen" in surrogate:
        return _chatml_action_regions(rendered_prompt)
    raise ValueError(f"No chat-template action extractor for surrogate model {model_name!r}")


def _action_step_token_indexes(
    messages: list[dict[Any, Any]], features: LogProbFeatures, model_name: str
) -> list[list[int]]:
    action_count = sum(message.get("role") == "assistant" for message in messages)
    if action_count == 0:
        raise ValueError("The prompt contains no assistant action messages")

    rendered_prompt = "".join(token or "" for token in features.tokens)
    action_regions = _action_regions(rendered_prompt, model_name)
    if len(action_regions) != action_count:
        raise ValueError(f"Expected {action_count} completed assistant action spans, found {len(action_regions)}")

    # then: align them to tokens
    #  1) divide the prompt length in characters into spans of each token
    token_spans: list[tuple[int, int]] = []
    token_start = 0
    for token in features.tokens:
        token_end = token_start + len(token or "")
        token_spans.append((token_start, token_end))
        token_start = token_end
    assert token_spans[0][0] == 0 and token_spans[-1][1] == len(rendered_prompt)
    for i in range(1, len(token_spans)):
        assert token_spans[i - 1][1] == token_spans[i][0]  # token spans partition the prompt

    # then: align them to tokens
    #  2) find the overlapping actions
    action_step_indexes = [
        [
            token_index  # grab the token index (equal to index position in `token_spans`)
            for token_index, (token_start, token_end) in enumerate(token_spans)
            # if the end of the token in rendered prompt is after the start of a region the action generated and the
            # start of the token is before the end of that same region
            if any(token_end > region_start and token_start < region_end for region_start, region_end in regions)
        ]
        # where each region is a character range in the rendered prompt that this action contributed
        for regions in action_regions
    ]
    if any(not indexes for indexes in action_step_indexes):
        raise ValueError("An assistant action span contains no tokens")
    return action_step_indexes


def _last_action_token_indexes(
    messages: list[dict[Any, Any]], features: LogProbFeatures, model_name: str
) -> list[int]:
    return _action_step_token_indexes(messages, features, model_name)[-1]


def _action_step_log_probs(
    messages: list[dict[Any, Any]], features: LogProbFeatures, model_name: str
) -> list[list[float]]:
    action_step_log_probs: list[list[float]] = []
    for action_step_token_indexes in _action_step_token_indexes(messages, features, model_name):
        log_probs: list[float] = []
        for index in action_step_token_indexes:
            log_prob = features.token_log_probs[index]
            if log_prob is None:
                raise ValueError(f"Assistant action token at index {index} has no log-probability")
            log_probs.append(log_prob)
        assert len(log_probs) == len(action_step_token_indexes)
        action_step_log_probs.append(log_probs)
    return action_step_log_probs


def _last_action_mean_log_prob(messages: list[dict[Any, Any]], features: LogProbFeatures, model_name: str) -> float:
    action_log_probs = _action_step_log_probs(messages, features, model_name)[-1]
    return math.fsum(action_log_probs) / len(action_log_probs)


def _aggregate_action_probability(
    messages: list[dict[Any, Any]], features: LogProbFeatures, aggregation_type: str, model_name: str
) -> float:
    # Each inner list contains the per-token log probabilities for one action step.
    action_steps_log_probs: list[list[float]] = _action_step_log_probs(messages, features, model_name)
    # log(prod_i(p_i)) = sum_i(log(p_i)) for each action step.
    sequence_log_probs: list[float] = [math.fsum(log_probs) for log_probs in action_steps_log_probs]
    # Dividing each sequence log probability by that action's token count gives its mean per-token log probability.
    length_normalized_log_probs = [
        sequence_log_prob / len(action_step_log_probs)
        for sequence_log_prob, action_step_log_probs in zip(sequence_log_probs, action_steps_log_probs, strict=True)
    ]

    match aggregation_type:
        case "last_action_mean_log_prob" | "len_norm_seq_prob_last":
            return math.exp(length_normalized_log_probs[-1])
        case "seq_prob_last":
            return math.exp(sequence_log_probs[-1])
        case "seq_prob_first":
            return math.exp(sequence_log_probs[0])
        case "seq_prob_mean":
            return math.exp(float(logsumexp(sequence_log_probs)) - math.log(len(sequence_log_probs)))
        case "seq_prob_min":
            return math.exp(min(sequence_log_probs))
        case "len_norm_seq_prob_first":
            return math.exp(length_normalized_log_probs[0])
        case "len_norm_seq_prob_mean":
            return math.exp(float(logsumexp(length_normalized_log_probs)) - math.log(len(length_normalized_log_probs)))
        case "len_norm_seq_prob_min":
            return math.exp(min(length_normalized_log_probs))
        case _:
            raise ValueError(f"Unsupported action log-probability aggregation: {aggregation_type!r}")


class LogProbsEstimator(BaseConfidenceEstimator):
    """Score the complete, OpenHands-formatted trajectory under a surrogate model."""

    def __init__(self, cfg: LogProbsEstimatorConfig, *, llm_limiter: LLMCallLimiter | None = None) -> None:
        self.cfg = cfg
        self.llm_limiter = llm_limiter
        self.logger = get_logger(self.__class__.__name__)
        self.replay_output_dir: Path | None = None
        if cfg.replay_from is not None:
            from crg_ce.estimators.openhands.config import output_dir_for_run_config

            self.replay_output_dir = output_dir_for_run_config(cfg.replay_from)

    def _prompt(self, ce_input: ConfEstimationInput) -> tuple[list[dict[Any, Any]], list[Any]]:
        state, events = load_conversation_state_and_events_from_archive(ce_input.conversation_archive_path)
        branch_events = get_active_branch_events(events, state)
        system_events = [event for event in branch_events if isinstance(event, SystemPromptEvent)]
        if len(system_events) != 1:
            raise ValueError(f"Expected exactly one OpenHands system prompt, found {len(system_events)}")

        llm = build_llm(self.cfg.agent)
        messages = trajectory_events_to_messages(branch_events)
        formatted_messages = llm.format_messages_for_llm(messages)
        tools = [tool.to_openai_tool(add_security_risk_prediction=True) for tool in system_events[0].tools]
        return formatted_messages, tools

    def _features_from_response(self, response: Any) -> LogProbFeatures:
        if len(response.choices) != 1:
            raise ValueError(f"Expected exactly one completion choice, received {len(response.choices)}")
        token_ids, tokens, token_log_probs = _prompt_token_features(response)
        scored_log_probs = [log_prob for log_prob in token_log_probs if log_prob is not None]
        if not scored_log_probs:
            raise ValueError("The response contains no scored prompt tokens")
        return LogProbFeatures(
            token_ids=token_ids,
            tokens=tokens,
            token_log_probs=token_log_probs,
            mean_log_prob=sum(scored_log_probs) / len(scored_log_probs),
        )

    def _completion_kwargs(self, messages: list[dict[Any, Any]], tools: list[Any]) -> dict[str, Any]:
        return {
            "model": self.cfg.agent.model_name,
            "messages": messages,
            "tools": tools,
            "api_key": resolve_api_key(self.cfg.agent.api_key),
            "base_url": self.cfg.agent.api_base,
            "timeout": self.cfg.agent.timeout,
            "allowed_openai_params": self.cfg.agent.allowed_openai_params,
            "max_completion_tokens": 1,
            "temperature": 0,
            "extra_body": {
                **self.cfg.agent.completion_kwargs,
                "prompt_logprobs": 0,
                "return_token_ids": True,
            },
        }

    def _openai_completion_kwargs(self, messages: list[dict[Any, Any]], tools: list[Any]) -> dict[str, Any]:
        return {
            "model": self.cfg.agent.model_name,
            "messages": messages,
            "tools": tools,
            "max_completion_tokens": 1,
            "reasoning_effort": self.cfg.agent.reasoning_effort,
            "timeout": self.cfg.agent.timeout,
            "extra_body": {
                **self.cfg.agent.completion_kwargs,
                "prompt_logprobs": 0,
                "return_token_ids": True,
            },
        }

    def _openai_client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        api_key = resolve_api_key(self.cfg.agent.api_key)
        if api_key is not None:
            kwargs["api_key"] = api_key
        if self.cfg.agent.api_base is not None:
            kwargs["base_url"] = self.cfg.agent.api_base
        return kwargs

    def _complete(self, messages: list[dict[Any, Any]], tools: list[Any]) -> Any:
        if self.cfg.completion_client == "litellm":
            return litellm.completion(**self._completion_kwargs(messages, tools))
        with OpenAI(**self._openai_client_kwargs()) as client:
            return client.chat.completions.create(**self._openai_completion_kwargs(messages, tools))

    async def _acomplete(self, messages: list[dict[Any, Any]], tools: list[Any]) -> Any:
        if self.cfg.completion_client == "litellm":
            return await litellm.acompletion(**self._completion_kwargs(messages, tools))
        async with AsyncOpenAI(**self._openai_client_kwargs()) as client:
            return await client.chat.completions.create(**self._openai_completion_kwargs(messages, tools))

    def _load_replay(self, ce_input: ConfEstimationInput) -> tuple[list[dict[Any, Any]], list[Any], LogProbFeatures]:
        if self.replay_output_dir is None:
            raise ValueError("Log-probability replay is not configured")
        source_output_dir = self.replay_output_dir / ce_input.instance_id / ce_input.model
        prompt_path = source_output_dir / "log_prob_prompt.json"
        features_path = source_output_dir / "log_prob_features.json"
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Replayed log-probability prompt does not exist: {prompt_path}")
        if not features_path.is_file():
            raise FileNotFoundError(f"Replayed log-probability features do not exist: {features_path}")
        prompt = json.loads(prompt_path.read_text())
        features = LogProbFeatures.model_validate_json(features_path.read_text())
        return prompt["messages"], prompt["tools"], features

    def _save_result(
        self,
        ce_input: ConfEstimationInput,
        messages: list[dict[Any, Any]],
        tools: list[Any],
        features: LogProbFeatures,
        stats: LiteLLMCallStats,
    ) -> ConfEstimationOutput:
        ce_input.output_dir.mkdir(exist_ok=True, parents=True)
        (ce_input.output_dir / "log_prob_features.json").write_text(features.model_dump_json(indent=2))
        (ce_input.output_dir / "log_prob_prompt.json").write_text(
            json.dumps({"messages": messages, "tools": tools}, indent=2)
        )
        confidence = (
            math.exp(features.mean_log_prob)
            if self.cfg.aggregation_type == "mean_log_prob"
            else _aggregate_action_probability(messages, features, self.cfg.aggregation_type, self.cfg.agent.model_name)
        )
        output = ConfEstimationOutput(
            confidence=confidence,
            total_tokens=stats.total_tokens,
            generated_tokens=stats.completion_tokens,
            cost=stats.cost,
            usage_by_model=model_usage_from_stats(stats),
        )
        self.logger.info(
            "Scored log-probability confidence of %s for %s",
            output.confidence,
            ce_input.instance_id,
        )
        self.save_output(output, ce_input.output_dir)
        return output

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        if self.replay_output_dir is not None:
            messages, tools, features = self._load_replay(ce_input)
            return self._save_result(ce_input, messages, tools, features, LiteLLMCallStats())

        messages, tools = self._prompt(ce_input)
        stats = LiteLLMCallStats()
        log_prompt(json.dumps({"messages": messages, "tools": tools}, indent=2), ce_input.output_dir)

        def complete_and_extract() -> LogProbFeatures:
            response = self._complete(messages, tools)
            stats.record_response(response, model=self.cfg.agent.model_name)
            return self._features_from_response(response)

        features = Retrying(stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception), reraise=True)(
            complete_and_extract
        )
        return self._save_result(ce_input, messages, tools, features, stats)

    async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        if self.replay_output_dir is not None:
            messages, tools, features = await asyncio.to_thread(self._load_replay, ce_input)
            return self._save_result(ce_input, messages, tools, features, LiteLLMCallStats())

        messages, tools = await asyncio.to_thread(self._prompt, ce_input)
        stats = LiteLLMCallStats()
        log_prompt(json.dumps({"messages": messages, "tools": tools}, indent=2), ce_input.output_dir)
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3), retry=retry_if_exception_type(Exception), reraise=True
        ):
            with attempt:
                if self.llm_limiter is not None:
                    async with self.llm_limiter.slot():
                        response = await self._acomplete(messages, tools)
                else:
                    response = await self._acomplete(messages, tools)
                stats.record_response(response, model=self.cfg.agent.model_name)
                features = self._features_from_response(response)
        return self._save_result(ce_input, messages, tools, features, stats)
