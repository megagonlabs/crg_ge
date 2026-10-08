import os
from dataclasses import dataclass, field
from typing import Any, cast

import litellm
from pydantic import BaseModel, SecretStr
from tenacity import AsyncRetrying, Retrying, retry_if_exception_type, stop_after_attempt

from crg_ce.llm_concurrency import LLMCallLimiter


@dataclass
class ModelLLMCallStats:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def merge(self, other: "ModelLLMCallStats") -> None:
        self.calls += other.calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.total_tokens += other.total_tokens
        self.cost += other.cost


@dataclass
class LLMStats:
    by_model: dict[str, ModelLLMCallStats] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(stats.calls for stats in self.by_model.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(stats.prompt_tokens for stats in self.by_model.values())

    @property
    def completion_tokens(self) -> int:
        return sum(stats.completion_tokens for stats in self.by_model.values())

    @property
    def reasoning_tokens(self) -> int:
        return sum(stats.reasoning_tokens for stats in self.by_model.values())

    @property
    def total_tokens(self) -> int:
        return sum(stats.total_tokens for stats in self.by_model.values())

    @property
    def cost(self) -> float:
        return sum(stats.cost for stats in self.by_model.values())

    def merge(self, other: "LLMStats") -> None:
        for model, other_model_stats in other.by_model.items():
            self.by_model.setdefault(model, ModelLLMCallStats()).merge(other_model_stats)

    def record_usage(
        self,
        *,
        model: str,
        calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        total_tokens: int,
        cost: float,
    ) -> None:
        self.by_model.setdefault(model, ModelLLMCallStats()).merge(
            ModelLLMCallStats(
                calls=calls,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                reasoning_tokens=reasoning_tokens,
                total_tokens=total_tokens,
                cost=cost,
            )
        )

    def record_response(self, response: Any, *, model: str | None = None) -> None:
        model_name = getattr(response, "model", None) or model
        if not model_name:
            raise ValueError("Cannot record LLM usage without a model name")

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage is not None else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage is not None else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage is not None else 0
        completion_tokens_details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        if isinstance(completion_tokens_details, dict):
            reasoning_tokens = completion_tokens_details.get("reasoning_tokens", 0)
        else:
            reasoning_tokens = getattr(completion_tokens_details, "reasoning_tokens", 0)

        response_cost = getattr(response, "response_cost", None)
        if response_cost is None:
            try:
                response_cost = litellm.completion_cost(completion_response=response)
            except Exception:
                response_cost = None
        if response_cost is not None:
            cost = float(response_cost)
        else:
            cost = 0.0

        self.record_usage(
            model=model_name,
            calls=1,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            reasoning_tokens=reasoning_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )


LiteLLMCallStats = LLMStats


class StructuredOutputError(ValueError):
    def __init__(self, message: str, *, content: object, tool_calls: object) -> None:
        super().__init__(message)
        self.content = content
        self.tool_calls = tool_calls


def complete_structured[StructuredOutputT: BaseModel](
    *,
    model: str,
    messages: list[dict[str, str]],
    output_model: type[StructuredOutputT],
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
) -> StructuredOutputT:
    output, _ = complete_structured_with_messages(
        model=model,
        messages=messages,
        output_model=output_model,
        api_key=api_key,
        base_url=base_url,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        allowed_openai_params=allowed_openai_params,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        stats=stats,
    )
    return output


async def acomplete_structured[StructuredOutputT: BaseModel](
    *,
    model: str,
    messages: list[dict[str, str]],
    output_model: type[StructuredOutputT],
    llm_limiter: LLMCallLimiter,
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
) -> StructuredOutputT:
    output, _ = await acomplete_structured_with_messages(
        model=model,
        messages=messages,
        output_model=output_model,
        llm_limiter=llm_limiter,
        api_key=api_key,
        base_url=base_url,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        allowed_openai_params=allowed_openai_params,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
        stats=stats,
    )
    return output


def tenaciously_complete_structured[StructuredOutputT: BaseModel](
    *,
    model: str,
    messages: list[dict[str, str]],
    output_model: type[StructuredOutputT],
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
    max_attempts: int = 3,
) -> StructuredOutputT:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    retryer = Retrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(StructuredOutputError),
        reraise=True,
    )
    return cast(
        StructuredOutputT,
        retryer(
            complete_structured,
            model=model,
            messages=messages,
            output_model=output_model,
            api_key=api_key,
            base_url=base_url,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            allowed_openai_params=allowed_openai_params,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
            stats=stats,
        ),
    )


async def atenaciously_complete_structured[StructuredOutputT: BaseModel](
    *,
    model: str,
    messages: list[dict[str, str]],
    output_model: type[StructuredOutputT],
    llm_limiter: LLMCallLimiter,
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
    max_attempts: int = 3,
) -> StructuredOutputT:
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(StructuredOutputError),
        reraise=True,
    ):
        with attempt:
            return await acomplete_structured(
                model=model,
                messages=messages,
                output_model=output_model,
                llm_limiter=llm_limiter,
                api_key=api_key,
                base_url=base_url,
                top_p=top_p,
                reasoning_effort=reasoning_effort,
                allowed_openai_params=allowed_openai_params,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
                stats=stats,
            )
    raise RuntimeError("AsyncRetrying completed without returning or raising")


def complete_structured_with_messages[StructuredOutputT: BaseModel](
    *,
    model: str,
    messages: list[dict[str, str]],
    output_model: type[StructuredOutputT],
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
) -> tuple[StructuredOutputT, list[dict[str, str]]]:
    response = litellm.completion(
        model=model,
        messages=messages,
        response_format=output_model,
        api_key=resolve_api_key(api_key),
        base_url=base_url,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        allowed_openai_params=allowed_openai_params,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
    )
    if stats is not None:
        stats.record_response(response, model=model)

    message = response.choices[0].message  # pyright: ignore[reportAttributeAccessIssue]
    content = message.content
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(content, str) or not content:
        raise StructuredOutputError(
            "LiteLLM response is missing structured output content",
            content=content,
            tool_calls=tool_calls,
        )
    try:
        output = output_model.model_validate_json(content)
    except ValueError as exc:
        raise StructuredOutputError(
            "LiteLLM response failed structured output validation",
            content=content,
            tool_calls=tool_calls,
        ) from exc
    return output, [*messages, {"role": "assistant", "content": content}]


async def acomplete_structured_with_messages[StructuredOutputT: BaseModel](
    *,
    model: str,
    messages: list[dict[str, str]],
    output_model: type[StructuredOutputT],
    llm_limiter: LLMCallLimiter,
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
) -> tuple[StructuredOutputT, list[dict[str, str]]]:
    async with llm_limiter.slot():
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            response_format=output_model,
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            allowed_openai_params=allowed_openai_params,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )
    if stats is not None:
        stats.record_response(response, model=model)

    message = response.choices[0].message  # pyright: ignore[reportAttributeAccessIssue]
    content = message.content
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(content, str) or not content:
        raise StructuredOutputError(
            "LiteLLM response is missing structured output content",
            content=content,
            tool_calls=tool_calls,
        )
    try:
        output = output_model.model_validate_json(content)
    except ValueError as exc:
        raise StructuredOutputError(
            "LiteLLM response failed structured output validation",
            content=content,
            tool_calls=tool_calls,
        ) from exc
    return output, [*messages, {"role": "assistant", "content": content}]


def complete_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
) -> str:
    response = litellm.completion(
        model=model,
        messages=messages,
        api_key=resolve_api_key(api_key),
        base_url=base_url,
        top_p=top_p,
        reasoning_effort=reasoning_effort,
        allowed_openai_params=allowed_openai_params,
        max_completion_tokens=max_completion_tokens,
        timeout=timeout,
    )
    if stats is not None:
        stats.record_response(response, model=model)

    content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
    if not isinstance(content, str) or not content:
        raise ValueError("LiteLLM response is missing text content")
    return content


async def acomplete_text(
    *,
    model: str,
    messages: list[dict[str, str]],
    llm_limiter: LLMCallLimiter,
    api_key: SecretStr | None = None,
    base_url: str | None = None,
    top_p: float | None = None,
    reasoning_effort: litellm.REASONING_EFFORT | None = None,
    allowed_openai_params: list[str] | None = None,
    max_completion_tokens: int | None = None,
    timeout: float | None = None,
    stats: LiteLLMCallStats | None = None,
) -> str:
    async with llm_limiter.slot():
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            top_p=top_p,
            reasoning_effort=reasoning_effort,
            allowed_openai_params=allowed_openai_params,
            max_completion_tokens=max_completion_tokens,
            timeout=timeout,
        )
    if stats is not None:
        stats.record_response(response, model=model)
    content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
    if not isinstance(content, str) or not content:
        raise ValueError("LiteLLM response is missing text content")
    return content


def resolve_api_key(api_key: SecretStr | None) -> str | None:
    if api_key is None:
        return None
    value = api_key.get_secret_value()
    if value.startswith("$"):
        env_var = value[1:]
        if env_var not in os.environ:
            raise ValueError(f"API key environment variable is not set: {env_var}")
        return os.environ[env_var]
    return value
