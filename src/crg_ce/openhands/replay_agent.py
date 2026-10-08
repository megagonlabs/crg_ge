import asyncio
import shutil
import tempfile
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Protocol, cast
from uuid import UUID

from openhands.sdk import Event, LocalWorkspace
from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.event import ActionEvent, ObservationBaseEvent
from openhands.sdk.io import InMemoryFileStore
from openhands.sdk.llm import Metrics

from crg_ce.estimators.openhands.config import AgentConfig, SupportedAgentToolSet
from crg_ce.llm_concurrency import LLMCallLimiter
from crg_ce.openhands.resume_points import ResumePoint
from crg_ce.utils.litellm_utils import LiteLLMCallStats
from crg_ce.utils.openhands import build_agent, import_needed_tools, replay_events
from crg_ce.utils.problem_logging import problem_log_context


class _RunnableConversation(Protocol):
    def run(self) -> None: ...


class _AsyncRunnableConversation(Protocol):
    async def arun(self) -> None: ...


class InvalidReplayedResponsesItemError(RuntimeError):
    """Raised when replayed history cannot be submitted to the Responses API."""


def normalize_overlong_tool_call_ids_for_responses(events: Sequence[Event]) -> Sequence[Event]:
    """Replace overlong replayed tool-call IDs with Responses-compatible IDs."""
    replacement_ids = {
        event.tool_call_id: f"fc_{sha256(event.tool_call_id.encode()).hexdigest()[:61]}"
        for event in events
        if isinstance(event, ActionEvent)
        and (len(event.tool_call_id) > 64 or len(event.tool_call.responses_item_id or "") > 64)
    }
    if not replacement_ids:
        return events

    normalized_events: list[Event] = []
    for event in events:
        if isinstance(event, ActionEvent) and event.tool_call_id in replacement_ids:
            tool_call_id = replacement_ids[event.tool_call_id]
            normalized_events.append(
                event.model_copy(
                    update={
                        "tool_call_id": tool_call_id,
                        "tool_call": event.tool_call.model_copy(update={"id": tool_call_id, "responses_item_id": None}),
                    }
                )
            )
        elif isinstance(event, ObservationBaseEvent) and event.tool_call_id in replacement_ids:
            normalized_events.append(event.model_copy(update={"tool_call_id": replacement_ids[event.tool_call_id]}))
        else:
            normalized_events.append(event)
    return normalized_events


@dataclass(frozen=True)
class ResumedAgentRun:
    """Inputs needed to resume a conversation prefix and run a new instruction."""

    agent_config: AgentConfig
    conversation_id: UUID
    events: Sequence[Event]
    resume_point: ResumePoint
    instruction: str
    max_steps: int
    workspace_dir: Path | None
    history_tool_set: SupportedAgentToolSet = "default"
    log_path: Path | None = None

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")


@dataclass(frozen=True)
class ResumedAgentRunResult:
    replayed_events: Sequence[Event]
    """The events from the prior conversation, replayed this conversation.run
    """
    generated_events: Sequence[Event]
    """The generated events from this conversation.run
    """
    llm_stats: LiteLLMCallStats = field(default_factory=LiteLLMCallStats)


@dataclass(frozen=True)
class ResumedAgentQuestionResult:
    response: str
    total_tokens: int = -1
    generated_tokens: int = -1
    cost: float = -1
    llm_stats: LiteLLMCallStats = field(default_factory=LiteLLMCallStats)


def get_new_llm_stats(
    conversation: LocalConversation,
    baseline_metrics: Metrics,
    model: str,
) -> LiteLLMCallStats:
    metrics = conversation.state.stats.get_combined_metrics().diff(baseline_metrics)
    usage = metrics.accumulated_token_usage
    stats = LiteLLMCallStats()
    if usage is None:
        return stats
    calls = len(metrics.token_usages)
    if (
        calls == 0
        and usage.prompt_tokens == 0
        and usage.completion_tokens == 0
        and usage.reasoning_tokens == 0
        and metrics.accumulated_cost == 0
    ):
        return stats
    stats.record_usage(
        model=model,
        calls=calls,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        total_tokens=usage.prompt_tokens + usage.completion_tokens,
        cost=metrics.accumulated_cost,
    )
    return stats


def run_resumed_agent(run: ResumedAgentRun) -> ResumedAgentRunResult:
    """Resume and run an OpenHands agent in the current process, returning a list of *generated* events."""

    log_context = problem_log_context(run.log_path) if run.log_path is not None else nullcontext()
    with log_context:
        import_needed_tools(run.history_tool_set)
        delete_workspace_dir: bool = False

        # if not set, default to a new temporary directory
        workspace_dir: Path = run.workspace_dir or Path(tempfile.mkdtemp())
        if not run.workspace_dir:  # if we created a tmp directory, remember to delete it
            delete_workspace_dir = True
        try:
            conversation = LocalConversation(
                agent=build_agent(run.agent_config),
                workspace=LocalWorkspace(working_dir=workspace_dir),
                persistence_dir=None,  # drop?
                file_store=InMemoryFileStore(),
                conversation_id=run.conversation_id,
                visualizer=None,
                delete_on_close=False,
                max_iteration_per_run=run.max_steps,
            )

            resumed_events = run.resume_point(run.events)
            replay_events(conversation, resumed_events)
            conversation.send_message(run.instruction)  # type: ignore

            before_run_event_count = len(conversation.state.events)
            metrics_before_run = conversation.state.stats.get_combined_metrics()
            cast(_RunnableConversation, conversation).run()
            # available_tools = [t.to_responses_tool() for t in conversation.agent.tools_map.values()]
            # Path("available_tools.json").write_text(json.dumps(available_tools, indent=2))

            return ResumedAgentRunResult(
                replayed_events=conversation.state.events[:before_run_event_count],
                generated_events=conversation.state.events[before_run_event_count:],
                llm_stats=get_new_llm_stats(conversation, metrics_before_run, run.agent_config.model_name),
            )
        finally:
            if delete_workspace_dir:
                shutil.rmtree(workspace_dir, ignore_errors=True)


async def arun_resumed_agent(
    run: ResumedAgentRun,
    *,
    llm_limiter: LLMCallLimiter,
) -> ResumedAgentRunResult:
    """Resume and run an OpenHands agent asynchronously in the current process."""

    log_context = problem_log_context(run.log_path) if run.log_path is not None else nullcontext()
    with log_context:
        import_needed_tools(run.history_tool_set)
        workspace_dir = run.workspace_dir or Path(tempfile.mkdtemp())
        delete_workspace_dir = run.workspace_dir is None
        try:
            conversation = LocalConversation(
                agent=build_agent(run.agent_config, llm_limiter=llm_limiter),
                workspace=LocalWorkspace(working_dir=workspace_dir),
                persistence_dir=None,
                file_store=InMemoryFileStore(),
                conversation_id=run.conversation_id,
                visualizer=None,
                delete_on_close=False,
                max_iteration_per_run=run.max_steps,
            )
            resumed_events = run.resume_point(run.events)
            replay_events(conversation, resumed_events)
            conversation.send_message(run.instruction)  # type: ignore

            before_run_event_count = len(conversation.state.events)
            metrics_before_run = conversation.state.stats.get_combined_metrics()
            await cast(_AsyncRunnableConversation, conversation).arun()
            return ResumedAgentRunResult(
                replayed_events=conversation.state.events[:before_run_event_count],
                generated_events=conversation.state.events[before_run_event_count:],
                llm_stats=get_new_llm_stats(conversation, metrics_before_run, run.agent_config.model_name),
            )
        finally:
            if delete_workspace_dir:
                await asyncio.to_thread(shutil.rmtree, workspace_dir, True)


def _run_resumed_agent_for_process(run: ResumedAgentRun) -> ResumedAgentRunResult:
    try:
        return run_resumed_agent(run)
    except Exception as error:
        raise RuntimeError(
            f"Resumed agent child process failed with {type(error).__name__}: {error}\n{traceback.format_exc()}"
        ) from None


def ask_question_in_resumed_run(run: ResumedAgentRun) -> ResumedAgentQuestionResult:
    """Resume an OpenHands agent and return its response with usage statistics."""

    log_context = problem_log_context(run.log_path) if run.log_path is not None else nullcontext()
    with log_context:
        import_needed_tools(run.history_tool_set)
        delete_workspace_dir: bool = False

        # if not set, default to a new temporary directory
        workspace_dir: Path = run.workspace_dir or Path(tempfile.mkdtemp())
        if not run.workspace_dir:  # if we created a tmp directory, remember to delete it
            delete_workspace_dir = True
        try:
            conversation = LocalConversation(
                agent=build_agent(run.agent_config),
                workspace=LocalWorkspace(working_dir=workspace_dir),
                persistence_dir=None,  # drop?
                file_store=InMemoryFileStore(),
                conversation_id=run.conversation_id,
                visualizer=None,
                delete_on_close=False,
                max_iteration_per_run=run.max_steps,
            )

            resumed_events = run.resume_point(run.events)
            if conversation.agent.llm.uses_responses_api():
                resumed_events = normalize_overlong_tool_call_ids_for_responses(resumed_events)
            replay_events(conversation, resumed_events)
            if conversation.agent.llm.uses_responses_api():
                messages = prepare_llm_messages(conversation.state.view)
                _, input_items = conversation.agent.llm.format_messages_for_responses(messages)
                for index, item in enumerate(input_items):
                    item_id = item.get("id")
                    if isinstance(item_id, str) and len(item_id) > 64:
                        raise InvalidReplayedResponsesItemError(
                            "Replayed OpenAI Responses input item has an overlong ID: "
                            f"input[{index}].id has length {len(item_id)} (type={item.get('type')!r}): {item!r}"
                        )
            metrics_before_ask = conversation.state.stats.get_combined_metrics()
            response: str = conversation.ask_agent(run.instruction)
            stats = get_new_llm_stats(conversation, metrics_before_ask, run.agent_config.model_name)
            usage_available = stats.prompt_tokens > 0 or stats.completion_tokens > 0
            if not usage_available:
                total_tokens = -1
                generated_tokens = -1
            else:
                # Provider completion/output tokens already include the reasoning-token subset.
                generated_tokens = stats.completion_tokens
                total_tokens = stats.total_tokens
            cost = stats.cost if stats.cost > 0 else -1
            return ResumedAgentQuestionResult(
                response=response,
                total_tokens=total_tokens,
                generated_tokens=generated_tokens,
                cost=cost,
                llm_stats=stats,
            )
        finally:
            if delete_workspace_dir:
                shutil.rmtree(workspace_dir, ignore_errors=True)


async def aask_question_in_resumed_run(
    run: ResumedAgentRun,
    *,
    llm_limiter: LLMCallLimiter,
) -> ResumedAgentQuestionResult:
    """Run the synchronous stateless OpenHands question off-loop under the LLM limit."""
    async with llm_limiter.slot():
        return await asyncio.to_thread(ask_question_in_resumed_run, run)


def run_single_resumed_agent_in_process(run: ResumedAgentRun) -> ResumedAgentRunResult:
    """Run a resumed OpenHands agent in a fresh child process."""

    with ProcessPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_resumed_agent_for_process, run)
        return future.result()
