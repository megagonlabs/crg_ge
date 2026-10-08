import asyncio

import pytest
from openhands.sdk import LLM as OpenHandsLLM

from crg_ce.estimators.openhands.config import AgentConfig
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.utils.openhands import build_llm


def test_openhands_llm_uses_configured_600_second_default_timeout() -> None:
    # This verifies all OpenHands-backed requests use the longer timeout by default while retaining explicit override.
    assert build_llm(AgentConfig(model_name="openai/test")).timeout == 600
    assert build_llm(AgentConfig(model_name="openai/test", timeout=45)).timeout == 45


def test_build_llm_forwards_allowed_openai_params() -> None:
    # This verifies agent-managed OpenHands calls preserve the explicit LiteLLM allowlist instead of nesting it in
    # extra_body, where LiteLLM would not use it for unsupported-parameter handling.
    llm = build_llm(AgentConfig(model_name="openai/test", allowed_openai_params=["reasoning_effort"]))

    prepared_kwargs = llm._prepare_transport_kwargs(messages=[], enable_streaming=False)

    assert prepared_kwargs["allowed_openai_params"] == ["reasoning_effort"]


def test_limiter_releases_permits_after_errors_and_cancellation() -> None:
    # This verifies failed and cancelled calls cannot leak the sole permit and deadlock later requests.
    async def exercise() -> None:
        limiter = LLMCallLimiter(1)
        with pytest.raises(RuntimeError, match="failed"):
            async with limiter.slot():
                raise RuntimeError("failed")

        entered = asyncio.Event()

        async def cancelled_call() -> None:
            async with limiter.slot():
                entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(cancelled_call())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        async with asyncio.timeout(1), limiter.slot():
            pass

    asyncio.run(exercise())


def test_openhands_completion_and_responses_share_the_limiter(monkeypatch) -> None:
    # This verifies both public async OpenHands transports use the same run-level limit across distinct LLM objects.
    active = 0
    maximum = 0

    async def fake_acompletion(self, *args, **kwargs):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            return "completion"
        finally:
            active -= 1

    async def fake_aresponses(self, *args, **kwargs):
        return await fake_acompletion(self, *args, **kwargs)

    monkeypatch.setattr(OpenHandsLLM, "acompletion", fake_acompletion)
    monkeypatch.setattr(OpenHandsLLM, "aresponses", fake_aresponses)

    async def exercise() -> None:
        limiter = LLMCallLimiter(1)
        first = build_llm(AgentConfig(model_name="openai/test"), llm_limiter=limiter)
        second = build_llm(AgentConfig(model_name="openai/test"), llm_limiter=limiter)
        await asyncio.gather(first.acompletion([]), second.aresponses([]))

    asyncio.run(exercise())
    assert maximum == 1
