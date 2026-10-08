import io
import tarfile
import uuid
from collections.abc import Sequence
from pathlib import Path

import pytest
from openhands.sdk import LLM, Agent, LocalConversation, LocalWorkspace
from openhands.sdk.agent.utils import prepare_llm_messages
from openhands.sdk.conversation import ConversationState
from openhands.sdk.conversation.persistence_const import BASE_STATE, EVENTS_DIR
from openhands.sdk.event import Event, MessageEvent
from openhands.sdk.llm import Message, TextContent
from openhands.tools.preset.default import get_default_tools

from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import (
    DivideAndConquerInterpTool,
    GatherEvidenceForAgenticGraphToolV1,
    ParticularizeInterpTool,
)
from crg_ce.utils.general import read_resource
from crg_ce.utils.openhands import (
    build_agent_tools,
    index_actions,
    load_conversation_state_and_events_from_archive,
    replay_events,
)

SAMPLE_TEST_CONVERSATION_TARGZ_PATH = Path(
    "src/crg_ce/utils/test_data/agronholm__typeguard.b6a7e438.combine_module__tr31kstl.tar.gz"
)
DUMMY_CONVERSATION_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def _agent() -> Agent:
    return Agent(llm=LLM(model="openai/test-model"), tools=[])


def _conversation(
    tmp_path: Path,
    *,
    persistence_dir: Path | None = None,
    conversation_id: uuid.UUID = DUMMY_CONVERSATION_ID,
) -> LocalConversation:
    return LocalConversation(
        agent=_agent(),
        workspace=LocalWorkspace(working_dir=tmp_path / "workspace"),
        persistence_dir=persistence_dir,
        conversation_id=conversation_id,
        visualizer=None,
        delete_on_close=False,
    )


def _conversation_from_state(state: ConversationState, tmp_path: Path) -> LocalConversation:
    return LocalConversation(
        agent=state.agent,
        workspace=LocalWorkspace(working_dir=tmp_path / "workspace"),
        persistence_dir=None,
        conversation_id=state.id,
        visualizer=None,
        delete_on_close=False,
    )


def _message_event(text: str, *, parent_id: str | None = None) -> MessageEvent:
    return MessageEvent(
        source="user",
        parent_id=parent_id,
        llm_message=Message(role="user", content=[TextContent(text=text)]),
    )


def _message_texts(messages: Sequence[Message]) -> list[str]:
    return [content.text for message in messages for content in message.content if isinstance(content, TextContent)]


def _view_texts(conversation: LocalConversation) -> list[str]:
    return _message_texts(prepare_llm_messages(conversation.state.view))


def test_build_agent_tools_uses_agent_config_preset() -> None:
    assert [tool.name for tool in build_agent_tools("default")] == [tool.name for tool in get_default_tools()]
    assert [tool.name for tool in build_agent_tools("gsn_agentic_gather_evidence_v1")] == [
        GatherEvidenceForAgenticGraphToolV1.name
    ]
    assert [tool.name for tool in build_agent_tools("gsn_agentic_graph_construction_interp")] == [
        DivideAndConquerInterpTool.name,
        ParticularizeInterpTool.name,
    ]


def _add_text_file(archive: tarfile.TarFile, name: str, text: str | bytes) -> None:
    data = text if isinstance(text, bytes) else text.encode()  # pyright: ignore[reportAttributeAccessIssue]
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def _archive_state_and_events(
    conversation: LocalConversation,
    archive_path: Path,
    events: Sequence[tuple[int, Event]],
) -> None:
    conversation_root = f"workspace/conversations/{conversation.state.id}"
    with tarfile.open(archive_path, "w:gz") as archive:
        _add_text_file(
            archive,
            f"{conversation_root}/{BASE_STATE}",
            conversation.state.model_dump_json(),
        )
        for index, event in events:
            _add_text_file(
                archive,
                f"{conversation_root}/{EVENTS_DIR}/event-{index:05d}-{event.id}.json",
                event.model_dump_json(exclude_none=True),
            )


def test_replay_events_updates_active_branch_for_parent_linked_events(tmp_path: Path):
    root = _message_event("root")
    child = _message_event("child", parent_id=root.id)
    events = [root, child]

    raw_append_conversation = _conversation(tmp_path / "raw")
    for event in events:
        raw_append_conversation.state.events.append(event)
    assert raw_append_conversation.state.active_branch() == []

    replay_conversation = _conversation(tmp_path / "replay")
    replay_events(replay_conversation, events)

    assert replay_conversation.state.active_branch() == events
    assert _view_texts(replay_conversation) == ["root", "child"]


def test_contents_of_archived_state():
    if not SAMPLE_TEST_CONVERSATION_TARGZ_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    state, events = load_conversation_state_and_events_from_archive(SAMPLE_TEST_CONVERSATION_TARGZ_PATH)
    assert state.last_user_message_id == "95ff7ecc-73d6-4687-8cb7-59b7b8b81063"
    last_user_message_event: Event = [e for e in events if e.id == state.last_user_message_id][0]
    assert last_user_message_event.kind == "MessageEvent"
    assert isinstance(last_user_message_event, MessageEvent)
    assert last_user_message_event.source == "user"
    last_user_message: Message = last_user_message_event.to_llm_message()
    assert last_user_message.role == "user"
    assert not last_user_message.tool_calls
    assert not last_user_message.tool_call_id
    assert not last_user_message.reasoning_content

    # find the first user message
    first_user_message: Message = [e for e in events if isinstance(e, MessageEvent) and e.source == "user"][
        0
    ].to_llm_message()
    assert first_user_message.role == "user"
    assert not first_user_message.tool_calls
    assert not first_user_message.tool_call_id
    assert not first_user_message.reasoning_content
    first_user_message_content: str = "".join(t.text for t in first_user_message.content if isinstance(t, TextContent))
    expected_first_user_message_content: str = read_resource(
        "utils/test_data/expected_first_user_message_content.agronholm.txt"
    )
    assert first_user_message_content == expected_first_user_message_content


def test_index_actions_from_real_archive_events() -> None:
    if not SAMPLE_TEST_CONVERSATION_TARGZ_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    _, events = load_conversation_state_and_events_from_archive(SAMPLE_TEST_CONVERSATION_TARGZ_PATH)
    partial_events = events[:10]

    actions_index = index_actions(partial_events)
    assert [(step, action.tool_name, len(observations)) for step, action, observations in actions_index.values()] == [
        (1, "think", 1),
        (2, "terminal", 1),
    ]


def test_replay_events_gets_active_branch_after_real_archive_partial_replay(tmp_path: Path):
    if not SAMPLE_TEST_CONVERSATION_TARGZ_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    state, events = load_conversation_state_and_events_from_archive(SAMPLE_TEST_CONVERSATION_TARGZ_PATH)
    assert state.id == uuid.UUID("a1bd0f7e-7e94-4fca-a1b5-00b03fd46159")
    assert len(events) == 206

    partial_events = events[:10]
    assert [type(event).__name__ for event in partial_events] == [
        "SystemPromptEvent",
        "MessageEvent",
        "ConversationStateUpdateEvent",
        "ConversationStateUpdateEvent",
        "ActionEvent",
        "ObservationEvent",
        "ConversationStateUpdateEvent",
        "ConversationStateUpdateEvent",
        "ActionEvent",
        "ObservationEvent",
    ]

    replay_conversation = _conversation_from_state(state, tmp_path / "replay")
    replay_events(replay_conversation, partial_events)

    active_branch = replay_conversation.state.active_branch()

    expected_active_branch_event_indexes = [0, 1, 4, 5, 8, 9]
    assert active_branch == [events[index] for index in expected_active_branch_event_indexes]
    assert [type(event).__name__ for event in active_branch] == [
        "SystemPromptEvent",
        "MessageEvent",
        "ActionEvent",
        "ObservationEvent",
        "ActionEvent",
        "ObservationEvent",
    ]


def test_full_replay(tmp_path: Path):
    # the most important test here: verify that our method of re-loading the events
    # has LLM-parity (same input text in messsages to the LLM) as if the conversation was resumed
    # with openhands (which has some other overhead we are attempting to avoid)
    if not SAMPLE_TEST_CONVERSATION_TARGZ_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    state, events = load_conversation_state_and_events_from_archive(SAMPLE_TEST_CONVERSATION_TARGZ_PATH)

    extract_root = tmp_path / "extracted"
    with tarfile.open(SAMPLE_TEST_CONVERSATION_TARGZ_PATH, "r:gz") as archive:
        archive.extractall(extract_root)

    expected_conversation = LocalConversation(
        agent=state.agent,
        workspace=LocalWorkspace(working_dir=tmp_path / "expected_workspace"),
        persistence_dir=extract_root / "workspace/conversations",
        conversation_id=state.id,
        max_iteration_per_run=state.max_iterations,
        stuck_detection=state.stuck_detection,
        visualizer=None,
        delete_on_close=False,
    )

    replay_conversation = _conversation_from_state(state, tmp_path / "replay")
    replay_events(replay_conversation, events)
    assert len(replay_conversation.state.events) > 10

    assert _view_texts(replay_conversation) == _view_texts(expected_conversation)


def test_replay_events_preserves_prepare_llm_messages_text(tmp_path: Path):
    source_conversation = _conversation(tmp_path / "source")
    source_events = [_message_event("first"), _message_event("second")]
    replay_events(source_conversation, source_events)
    expected_texts = _view_texts(source_conversation)

    replay_conversation = _conversation(tmp_path / "target")
    replay_events(replay_conversation, source_conversation.state.active_branch())

    assert _view_texts(replay_conversation) == expected_texts


def test_load_conversation_state_and_events_from_archive(tmp_path: Path):
    conversation = _conversation(tmp_path / "conversation")
    events = [_message_event("first"), _message_event("second")]
    archive_path = tmp_path / "conversation.tar.gz"
    _archive_state_and_events(
        conversation,
        archive_path,
        [(index, event) for index, event in enumerate(events)],
    )

    loaded_state, loaded_events = load_conversation_state_and_events_from_archive(archive_path)

    assert loaded_state.id == conversation.state.id
    assert [event.id for event in loaded_events] == [event.id for event in events]


def test_load_conversation_state_and_events_from_archive_rejects_multiple_conversations(tmp_path: Path):
    # This verifies archives bundling several conversations fail instead of silently selecting one; it assumes each
    # complete conversation includes its own base_state.json.
    conversation = _conversation(tmp_path / "selected")
    selected_events = [_message_event("selected")]
    other_conversation = _conversation(tmp_path / "other", conversation_id=uuid.uuid4())
    other_event = _message_event("other")
    archive_path = tmp_path / "conversations.tar.gz"
    selected_root = f"workspace/conversations/{conversation.state.id}"
    other_root = f"workspace/conversations/{other_conversation.state.id}"

    with tarfile.open(archive_path, "w:gz") as archive:
        _add_text_file(archive, f"{selected_root}/{BASE_STATE}", conversation.state.model_dump_json())
        _add_text_file(
            archive,
            f"{selected_root}/{EVENTS_DIR}/event-00000-{selected_events[0].id}.json",
            selected_events[0].model_dump_json(exclude_none=True),
        )
        _add_text_file(archive, f"{other_root}/{BASE_STATE}", other_conversation.state.model_dump_json())
        _add_text_file(
            archive,
            f"{other_root}/{EVENTS_DIR}/event-00000-{other_event.id}.json",
            other_event.model_dump_json(exclude_none=True),
        )

    with pytest.raises(ValueError, match="contains multiple conversations"):
        load_conversation_state_and_events_from_archive(archive_path)


def test_load_conversation_state_and_events_from_archive_rejects_event_index_gaps(tmp_path: Path):
    # verifies missing event handled
    conversation = _conversation(tmp_path / "conversation")
    archive_path = tmp_path / "conversation.tar.gz"
    _archive_state_and_events(
        conversation,
        archive_path,
        [(1, _message_event("gap"))],
    )

    with pytest.raises(ValueError, match="Event index gap"):
        load_conversation_state_and_events_from_archive(archive_path)


def test_load_conversation_state_and_events_from_archive_rejects_payload_id_mismatch(tmp_path: Path):
    # verifies a malformed event sequence
    conversation = _conversation(tmp_path / "conversation")
    event = _message_event("event")
    mismatched_event = event.model_copy(update={"id": str(uuid.uuid4())})
    archive_path = tmp_path / "conversation.tar.gz"
    conversation_root = f"workspace/conversations/{conversation.state.id}"

    with tarfile.open(archive_path, "w:gz") as archive:
        _add_text_file(
            archive,
            f"{conversation_root}/{BASE_STATE}",
            conversation.state.model_dump_json(),
        )
        _add_text_file(
            archive,
            f"{conversation_root}/{EVENTS_DIR}/event-00000-{event.id}.json",
            mismatched_event.model_dump_json(exclude_none=True),
        )

    with pytest.raises(ValueError, match="does not match filename id"):
        load_conversation_state_and_events_from_archive(archive_path)
