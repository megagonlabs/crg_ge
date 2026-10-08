import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Protocol


class LLMConcurrencyConfig(Protocol):
    max_concurrent_llm_calls: int | None


class LLMCallLimiter:
    """Run-scoped, re-entrant limit for concurrent logical LLM calls."""

    def __init__(self, max_concurrent_calls: int) -> None:
        if max_concurrent_calls <= 0:
            raise ValueError("max_concurrent_calls must be positive")
        self.max_concurrent_calls = max_concurrent_calls
        self._semaphore = asyncio.Semaphore(max_concurrent_calls)
        self._depth: ContextVar[int] = ContextVar(f"llm_call_limiter_depth_{id(self)}", default=0)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        depth = self._depth.get()
        if depth:
            token = self._depth.set(depth + 1)
            try:
                yield
            finally:
                self._depth.reset(token)
            return

        async with self._semaphore:
            token = self._depth.set(1)
            try:
                yield
            finally:
                self._depth.reset(token)


def build_llm_limiters(agent_configs: Sequence[LLMConcurrencyConfig]) -> dict[int, LLMCallLimiter]:
    """Create one run-scoped limiter for each configured agent instance."""
    return {
        id(agent_config): LLMCallLimiter(agent_config.max_concurrent_llm_calls)
        for agent_config in agent_configs
        if agent_config.max_concurrent_llm_calls is not None
    }
