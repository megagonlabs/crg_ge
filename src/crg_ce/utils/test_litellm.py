from types import SimpleNamespace

import pytest
from litellm.types.utils import Choices, ModelResponse
from litellm.types.utils import Message as LitellmMessage
from pydantic import BaseModel, SecretStr

from crg_ce.utils.litellm_utils import (
    LiteLLMCallStats,
    complete_structured,
    complete_text,
    resolve_api_key,
    tenaciously_complete_structured,
)


class _StructuredOutput(BaseModel):
    value: str


def _model_response(content: str | None) -> ModelResponse:
    return ModelResponse(
        id="response-1",
        created=0,
        model="test-model",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LitellmMessage(
                    content=content,
                    role="assistant",
                    tool_calls=None,
                    function_call=None,
                ),
            )
        ],
    )


def test_complete_structured_parses_model_response(monkeypatch) -> None:
    expected = _StructuredOutput(value="parsed")

    monkeypatch.setattr(
        "crg_ce.utils.litellm_utils.litellm.completion",
        lambda **kwargs: _model_response(expected.model_dump_json()),
    )

    assert (
        complete_structured(
            model="test-model",
            messages=[{"role": "user", "content": "return structured output"}],
            output_model=_StructuredOutput,
        )
        == expected
    )


def test_complete_structured_rejects_missing_content(monkeypatch) -> None:
    monkeypatch.setattr(
        "crg_ce.utils.litellm_utils.litellm.completion",
        lambda **kwargs: _model_response(None),
    )

    with pytest.raises(ValueError, match="missing structured output content"):
        complete_structured(
            model="test-model",
            messages=[{"role": "user", "content": "return structured output"}],
            output_model=_StructuredOutput,
        )


def test_tenaciously_complete_structured_retries_invalid_structure(monkeypatch) -> None:
    # This verifies malformed structured responses are retried up to the configured attempt limit.
    # It assumes a later response that validates should be returned without another retry.
    expected = _StructuredOutput(value="parsed")
    responses = iter([_model_response(None), _model_response("{}"), _model_response(expected.model_dump_json())])
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    monkeypatch.setattr("crg_ce.utils.litellm_utils.litellm.completion", fake_completion)

    result = tenaciously_complete_structured(
        model="test-model",
        messages=[{"role": "user", "content": "return structured output"}],
        output_model=_StructuredOutput,
    )

    assert result == expected
    assert len(calls) == 3


def test_complete_text_returns_model_response_content(monkeypatch) -> None:
    # This verifies unstructured LiteLLM calls expose the assistant text used by ask-and-parse estimators.
    monkeypatch.setattr(
        "crg_ce.utils.litellm_utils.litellm.completion",
        lambda **kwargs: _model_response("Reasoning\nConfidence: 75%"),
    )

    assert complete_text(model="test-model", messages=[{"role": "user", "content": "estimate"}]) == (
        "Reasoning\nConfidence: 75%"
    )


def test_complete_text_forwards_allowed_openai_params(monkeypatch) -> None:
    # This verifies an explicitly configured OpenAI-compatible parameter allowlist reaches LiteLLM unchanged, so
    # provider metadata cannot discard an otherwise valid custom endpoint parameter.
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _model_response("ok")

    monkeypatch.setattr("crg_ce.utils.litellm_utils.litellm.completion", fake_completion)

    complete_text(
        model="openai/test-model",
        messages=[{"role": "user", "content": "estimate"}],
        allowed_openai_params=["reasoning_effort"],
    )

    assert calls[0]["allowed_openai_params"] == ["reasoning_effort"]


def test_complete_text_forwards_timeout(monkeypatch) -> None:
    # This verifies the per-agent timeout reaches LiteLLM instead of silently using LiteLLM's global fallback.
    calls = []

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return _model_response("ok")

    monkeypatch.setattr("crg_ce.utils.litellm_utils.litellm.completion", fake_completion)

    complete_text(
        model="openai/test-model",
        messages=[{"role": "user", "content": "estimate"}],
        timeout=1200,
    )

    assert calls[0]["timeout"] == 1200


def test_complete_structured_records_response_stats(monkeypatch) -> None:
    expected = _StructuredOutput(value="parsed")
    response = SimpleNamespace(
        model="resolved-test-model",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=expected.model_dump_json()),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=5,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=2),
            total_tokens=8,
        ),
        response_cost=0.125,
    )
    stats = LiteLLMCallStats()

    monkeypatch.setattr(
        "crg_ce.utils.litellm_utils.litellm.completion",
        lambda **kwargs: response,
    )

    assert (
        complete_structured(
            model="test-model",
            messages=[{"role": "user", "content": "return structured output"}],
            output_model=_StructuredOutput,
            stats=stats,
        )
        == expected
    )
    assert stats.calls == 1
    assert stats.prompt_tokens == 3
    assert stats.completion_tokens == 5
    assert stats.reasoning_tokens == 2
    assert stats.total_tokens == 8
    assert stats.cost == 0.125
    assert stats.by_model["resolved-test-model"].calls == 1
    assert stats.by_model["resolved-test-model"].total_tokens == 8


def test_complete_structured_records_reasoning_tokens_from_dict_details(monkeypatch) -> None:
    expected = _StructuredOutput(value="parsed")
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=expected.model_dump_json()),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=3,
            completion_tokens=5,
            completion_tokens_details={"reasoning_tokens": 4},
            total_tokens=8,
        ),
        response_cost=0.125,
    )
    stats = LiteLLMCallStats()

    monkeypatch.setattr(
        "crg_ce.utils.litellm_utils.litellm.completion",
        lambda **kwargs: response,
    )

    complete_structured(
        model="test-model",
        messages=[{"role": "user", "content": "return structured output"}],
        output_model=_StructuredOutput,
        stats=stats,
    )

    assert stats.reasoning_tokens == 4


def test_resolve_api_key_reads_environment_variable(monkeypatch) -> None:
    monkeypatch.setenv("TEST_LITELLM_API_KEY", "secret")

    assert resolve_api_key(SecretStr("$TEST_LITELLM_API_KEY")) == "secret"


def test_resolve_api_key_rejects_missing_environment_variable() -> None:
    with pytest.raises(ValueError, match="API key environment variable is not set"):
        resolve_api_key(SecretStr("$MISSING_LITELLM_API_KEY"))
