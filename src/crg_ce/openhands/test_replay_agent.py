import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from openhands.sdk import Message, TextContent
from openhands.sdk.conversation.conversation_stats import ConversationStats
from openhands.sdk.event import ActionEvent, AgentErrorEvent, MessageEvent
from openhands.sdk.llm import MessageToolCall, Metrics

from crg_ce.estimators.openhands.config import AgentConfig
from crg_ce.openhands.replay_agent import (
    InvalidReplayedResponsesItemError,
    ResumedAgentRun,
    ResumedAgentRunResult,
    _run_resumed_agent_for_process,
    ask_question_in_resumed_run,
    normalize_overlong_tool_call_ids_for_responses,
    run_resumed_agent,
    run_single_resumed_agent_in_process,
)

DUMMY_CONVERSATION_ID = UUID("00000000-0000-0000-0000-000000000000")


class _FakeConversation:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.state = SimpleNamespace(events=[], max_iterations=None, stats=ConversationStats(), view=object())
        self.agent = SimpleNamespace(
            tools_map={},
            llm=SimpleNamespace(uses_responses_api=lambda: False),
        )
        self.sent_messages: list[str] = []
        self.asked_questions: list[str] = []
        self.ask_metrics: Metrics | None = None
        self.run_metrics: Metrics | None = None

    def send_message(self, instruction: str) -> None:
        self.sent_messages.append(instruction)
        self.state.events.append(
            MessageEvent(
                source="user",
                llm_message=Message(role="user", content=[TextContent(text=instruction)]),
            )
        )

    def run(self) -> None:
        if self.run_metrics is not None:
            metrics = self.state.stats.usage_to_metrics.setdefault("default", Metrics(model_name="openai/test-model"))
            metrics.merge(self.run_metrics)
        self.state.events.append(
            MessageEvent(
                source="agent",
                llm_message=Message(role="assistant", content=[TextContent(text="generated")]),
            )
        )

    def ask_agent(self, instruction: str) -> str:
        self.asked_questions.append(instruction)
        if self.ask_metrics is not None:
            metrics = self.state.stats.usage_to_metrics.setdefault("default", Metrics(model_name="openai/test-model"))
            metrics.merge(self.ask_metrics)
        return "Confidence: 75%"


def _message(text: str) -> MessageEvent:
    return MessageEvent(source="user", llm_message=Message(role="user", content=[TextContent(text=text)]))


def test_normalize_overlong_tool_call_ids_for_responses_rewrites_tool_call_references() -> None:
    # This guarantees overlong tool-call IDs are replaced and their tool-result references stay linked.
    original_id = "call_123__thought__" + "signature" * 10
    action = ActionEvent(
        thought=[],
        action=None,
        tool_name="terminal",
        tool_call_id=original_id,
        tool_call=MessageToolCall(id=original_id, name="terminal", arguments="{}", origin="completion"),
        llm_response_id="response_123",
    )
    observation = AgentErrorEvent(error="tool failed", tool_name="terminal", tool_call_id=original_id)

    normalized = normalize_overlong_tool_call_ids_for_responses([action, observation])

    normalized_action = normalized[0]
    normalized_observation = normalized[1]
    assert isinstance(normalized_action, ActionEvent)
    assert isinstance(normalized_observation, AgentErrorEvent)
    assert normalized_action.tool_call_id.startswith("fc_")
    assert len(normalized_action.tool_call_id) == 64
    assert normalized_action.tool_call.id == normalized_action.tool_call_id
    assert normalized_action.tool_call.responses_item_id is None
    assert normalized_action.tool_call.to_responses_dict()["id"] == normalized_action.tool_call_id
    assert normalized_observation.tool_call_id == normalized_action.tool_call_id
    assert action.tool_call_id == original_id
    assert observation.tool_call_id == original_id


def test_normalize_overlong_tool_call_ids_for_responses_leaves_other_events_unchanged() -> None:
    # This guarantees valid tool-call IDs preserve their original event objects and replay behavior.
    action = ActionEvent(
        thought=[],
        action=None,
        tool_name="terminal",
        tool_call_id="call_123",
        tool_call=MessageToolCall(id="call_123", name="terminal", arguments="{}", origin="completion"),
        llm_response_id="response_123",
    )
    events = [action]

    assert normalize_overlong_tool_call_ids_for_responses(events) is events


def test_run_resumed_agent_replays_selected_events_and_returns_generated_events(tmp_path: Path, monkeypatch) -> None:
    # This verifies replay setup and generated-event slicing, assuming callers provide a stable conversation id.
    source_events = [_message("keep"), _message("drop")]
    selected_events = [source_events[0]]
    conversations: list[_FakeConversation] = []

    def fake_local_conversation(**kwargs):
        conversation = _FakeConversation(**kwargs)
        conversations.append(conversation)
        return conversation

    def fake_replay_events(conversation: _FakeConversation, events):
        conversation.state.events.extend(events)

    imported_tool_sets: list[str] = []
    monkeypatch.setattr("crg_ce.openhands.replay_agent.import_needed_tools", imported_tool_sets.append)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.build_agent", lambda cfg: ("agent", cfg))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalWorkspace", lambda working_dir: ("workspace", working_dir))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalConversation", fake_local_conversation)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.replay_events", fake_replay_events)
    result = run_resumed_agent(
        ResumedAgentRun(
            agent_config=AgentConfig(model_name="openai/test-model"),
            conversation_id=DUMMY_CONVERSATION_ID,
            events=source_events,
            resume_point=lambda events: selected_events,
            instruction="new instruction",
            max_steps=3,
            workspace_dir=tmp_path / "workspace",
        )
    )

    assert imported_tool_sets == ["default"]
    assert len(conversations) == 1
    assert conversations[0].state.events[:1] == selected_events
    assert conversations[0].sent_messages == ["new instruction"]
    assert conversations[0].kwargs["conversation_id"] == DUMMY_CONVERSATION_ID
    assert conversations[0].kwargs["max_iteration_per_run"] == 3
    assert [event.llm_message.content[0].text for event in result.replayed_events] == [  # type: ignore
        "keep",
        "new instruction",
    ]
    assert [event.llm_message.content[0].text for event in result.generated_events] == ["generated"]  # type: ignore


def test_run_resumed_agent_reports_only_usage_added_after_replay(tmp_path: Path, monkeypatch) -> None:
    # This verifies archived conversation usage is excluded while every new resumed-run token and cost is reported.
    conversations: list[_FakeConversation] = []

    def metrics(prompt_tokens: int, completion_tokens: int, cost: float, response_id: str) -> Metrics:
        result = Metrics(model_name="openai/test-model")
        result.add_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=completion_tokens // 2,
            context_window=1000,
            response_id=response_id,
        )
        result.add_cost(cost)
        return result

    def fake_local_conversation(**kwargs):
        conversation = _FakeConversation(**kwargs)
        conversation.run_metrics = metrics(80, 20, 0.05, "new-response")
        conversations.append(conversation)
        return conversation

    def fake_replay_events(conversation: _FakeConversation, events) -> None:
        conversation.state.stats.usage_to_metrics["default"] = metrics(800, 200, 0.5, "archived-response")

    monkeypatch.setattr("crg_ce.openhands.replay_agent.import_needed_tools", lambda tool_set: None)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.build_agent", lambda cfg: ("agent", cfg))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalWorkspace", lambda working_dir: ("workspace", working_dir))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalConversation", fake_local_conversation)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.replay_events", fake_replay_events)
    result = run_resumed_agent(
        ResumedAgentRun(
            agent_config=AgentConfig(model_name="openai/test-model"),
            conversation_id=DUMMY_CONVERSATION_ID,
            events=[],
            resume_point=lambda events: [],
            instruction="new instruction",
            max_steps=1,
            workspace_dir=tmp_path / "workspace",
        )
    )

    assert len(conversations) == 1
    assert result.llm_stats.calls == 1
    assert result.llm_stats.prompt_tokens == 80
    assert result.llm_stats.completion_tokens == 20
    assert result.llm_stats.reasoning_tokens == 10
    assert result.llm_stats.total_tokens == 100
    assert result.llm_stats.cost == pytest.approx(0.05)
    assert result.llm_stats.by_model["openai/test-model"].calls == 1
    assert result.llm_stats.by_model["openai/test-model"].total_tokens == 100


def test_run_resumed_agent_writes_problem_log_when_requested(tmp_path: Path, monkeypatch) -> None:
    # This verifies replay-agent child-process entrypoints can attach logs to the caller's per-problem log file.
    conversations: list[_FakeConversation] = []
    log_path = tmp_path / "openhands_gsn.log"

    def fake_local_conversation(**kwargs):
        conversation = _FakeConversation(**kwargs)
        conversations.append(conversation)
        return conversation

    def fake_replay_events(conversation: _FakeConversation, events):
        logging.getLogger("crg_ce.tests.replay_agent").warning("replay log marker")

    monkeypatch.setattr("crg_ce.openhands.replay_agent.import_needed_tools", lambda tool_set: None)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.build_agent", lambda cfg: ("agent", cfg))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalWorkspace", lambda working_dir: ("workspace", working_dir))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalConversation", fake_local_conversation)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.replay_events", fake_replay_events)
    run_resumed_agent(
        ResumedAgentRun(
            agent_config=AgentConfig(model_name="openai/test-model"),
            conversation_id=DUMMY_CONVERSATION_ID,
            events=[],
            resume_point=lambda events: [],
            instruction="new instruction",
            max_steps=1,
            workspace_dir=tmp_path / "workspace",
            log_path=log_path,
        )
    )

    assert len(conversations) == 1
    assert "replay log marker" in log_path.read_text()


def test_ask_question_in_resumed_run_returns_response_usage_and_cost(tmp_path: Path, monkeypatch) -> None:
    # This verifies resumed questions expose provider totals without double-counting reasoning tokens.
    conversations: list[_FakeConversation] = []

    def fake_local_conversation(**kwargs):
        conversation = _FakeConversation(**kwargs)
        metrics = Metrics(model_name="openai/test-model")
        metrics.add_token_usage(
            prompt_tokens=80,
            completion_tokens=20,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=12,
            context_window=1000,
            response_id="response-id",
        )
        metrics.add_cost(0.05)
        conversation.ask_metrics = metrics
        conversations.append(conversation)
        return conversation

    monkeypatch.setattr("crg_ce.openhands.replay_agent.import_needed_tools", lambda tool_set: None)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.build_agent", lambda cfg: ("agent", cfg))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalWorkspace", lambda working_dir: ("workspace", working_dir))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalConversation", fake_local_conversation)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.replay_events", lambda conversation, events: None)
    result = ask_question_in_resumed_run(
        ResumedAgentRun(
            agent_config=AgentConfig(model_name="openai/test-model"),
            conversation_id=DUMMY_CONVERSATION_ID,
            events=[],
            resume_point=lambda events: [],
            instruction="confidence instruction",
            max_steps=1,
            workspace_dir=tmp_path / "workspace",
        )
    )

    assert result.response == "Confidence: 75%"
    assert result.total_tokens == 100
    assert result.generated_tokens == 20
    assert result.cost == 0.05
    assert conversations[0].asked_questions == ["confidence instruction"]


def test_ask_question_in_resumed_run_rejects_overlong_responses_item_ids(tmp_path: Path, monkeypatch) -> None:
    # This guarantees invalid historical Responses item IDs fail before any provider request is made.
    conversation = _FakeConversation()
    conversation.agent.llm = SimpleNamespace(
        uses_responses_api=lambda: True,
        format_messages_for_responses=lambda messages: (None, [{"type": "function_call", "id": "x" * 65}]),
    )

    monkeypatch.setattr("crg_ce.openhands.replay_agent.import_needed_tools", lambda tool_set: None)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.build_agent", lambda cfg: ("agent", cfg))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalWorkspace", lambda working_dir: ("workspace", working_dir))
    monkeypatch.setattr("crg_ce.openhands.replay_agent.LocalConversation", lambda **kwargs: conversation)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.replay_events", lambda conversation, events: None)
    monkeypatch.setattr("crg_ce.openhands.replay_agent.prepare_llm_messages", lambda view: [])
    with pytest.raises(InvalidReplayedResponsesItemError, match=r"input\[0\]\.id has length 65"):
        ask_question_in_resumed_run(
            ResumedAgentRun(
                agent_config=AgentConfig(model_name="openai/test-model"),
                conversation_id=DUMMY_CONVERSATION_ID,
                events=[],
                resume_point=lambda events: [],
                instruction="confidence instruction",
                max_steps=1,
                workspace_dir=tmp_path / "workspace",
            )
        )

    assert conversation.asked_questions == []


def test_run_resumed_agent_in_process_uses_child_executor(monkeypatch, tmp_path: Path) -> None:
    # This verifies process isolation delegates the exact replay request, including its dummy conversation id.
    generated = [_message("generated")]
    submitted_runs: list[ResumedAgentRun] = []

    class FakeFuture:
        def __init__(self, replayed_events) -> None:
            self.replayed_events = replayed_events
            pass

        def result(self):
            return ResumedAgentRunResult(replayed_events=self.replayed_events, generated_events=generated)

    class FakeExecutor:
        def __init__(self, max_workers: int) -> None:
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            pass

        def submit(self, fn, run: ResumedAgentRun):
            submitted_runs.append(run)
            assert fn.__name__ == "_run_resumed_agent_for_process"
            assert self.max_workers == 1
            return FakeFuture(run.events)

    monkeypatch.setattr("crg_ce.openhands.replay_agent.ProcessPoolExecutor", FakeExecutor)

    run = ResumedAgentRun(
        agent_config=AgentConfig(model_name="openai/test-model"),
        conversation_id=DUMMY_CONVERSATION_ID,
        events=[],
        resume_point=lambda events: [],
        instruction="new instruction",
        max_steps=1,
        workspace_dir=tmp_path / "workspace",
    )

    result = run_single_resumed_agent_in_process(run)

    assert result.replayed_events == []
    assert result.generated_events == generated
    assert submitted_runs == [run]


def test_process_wrapper_transports_original_exception_as_picklable_traceback(monkeypatch, tmp_path: Path) -> None:
    # This guarantees child failures retain their original type, message, and traceback across process serialization.
    def fail_run(_run: ResumedAgentRun) -> ResumedAgentRunResult:
        raise ValueError("provider rejected request")

    monkeypatch.setattr("crg_ce.openhands.replay_agent.run_resumed_agent", fail_run)
    run = ResumedAgentRun(
        agent_config=AgentConfig(model_name="openai/test-model"),
        conversation_id=DUMMY_CONVERSATION_ID,
        events=[],
        resume_point=lambda events: [],
        instruction="new instruction",
        max_steps=1,
        workspace_dir=tmp_path / "workspace",
    )

    with pytest.raises(RuntimeError) as exc_info:
        _run_resumed_agent_for_process(run)

    transported_error = pickle.loads(pickle.dumps(exc_info.value))
    assert "ValueError: provider rejected request" in str(transported_error)
    assert "fail_run" in str(transported_error)
