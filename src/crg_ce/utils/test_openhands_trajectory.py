from collections import Counter
from pathlib import Path

import pytest
from openhands.sdk import LocalWorkspace, Message, TextContent
from openhands.sdk.conversation import LocalConversation
from openhands.sdk.event import ACPToolCallEvent, ActionEvent, MessageEvent, ObservationEvent

from crg_ce.utils.openhands import load_conversation_state_and_events_from_archive, replay_events
from crg_ce.utils.openhands_trajectory import (
    TRUNCATED_CONTENT_NOTE,
    ACPToolCall,
    ACPUserMessage,
    ArchivedACPAction,
    SummarizedACPActionEvent,
    SummarizedACPObservationEvent,
    TitleInferredFileEditorAction,
    acp_messages_to_openhands_events,
    condense_uncited_action_steps,
    get_active_branch_events,
    render_acp_trajectory,
    render_trajectory,
)

ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/agronholm__typeguard.b6a7e438.combine_module__tr31kstl.tar.gz")
ASTROPY_ACP_ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/astropy__astropy-7166.codex.tar.gz")
ASTROPY_ACP_RENDERED_FIXTURE_PATH = Path("src/crg_ce/utils/test_data/astropy__astropy-7166.codex.rendered.xml")
ASTROPY_CLAUDE_CODE_ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/astropy__astropy-7166.claude-code.tar.gz")
ASTROPY_CLAUDE_CODE_RENDERED_FIXTURE_PATH = Path(
    "src/crg_ce/utils/test_data/astropy__astropy-7166.claude-code.rendered.xml"
)


def _assert_claude_code_acp_observation_content_schema(archive_path: Path) -> Counter[str]:
    state, events = load_conversation_state_and_events_from_archive(archive_path)
    terminal_events = [
        event
        for event in get_active_branch_events(events, state)
        if isinstance(event, ACPToolCallEvent) and event.status in {"completed", "failed"}
    ]
    assert terminal_events

    block_types: Counter[str] = Counter()
    for event in terminal_events:
        assert event.content is not None
        assert len(event.content) == 1
        block = event.model_dump(mode="python")["content"][0]
        assert block["field_meta"] is None
        block_types[block["type"]] += 1

        if block["type"] == "content":
            assert set(block) == {"content", "field_meta", "type"}
            inner = block["content"]
            assert set(inner) == {"annotations", "field_meta", "text", "type"}
            assert inner["type"] == "text"
            assert inner["annotations"] is None
            assert inner["field_meta"] is None
            assert isinstance(inner["text"], str)
        elif block["type"] == "diff":
            assert set(block) == {"field_meta", "new_text", "old_text", "path", "type"}
            assert all(isinstance(block[key], str) for key in ("new_text", "old_text", "path"))
        else:
            raise AssertionError(f"Unexpected ACP observation content block type: {block['type']}")

    return block_types


def test_acp_normalization_coalesces_completed_then_pending_tool_call_snapshot() -> None:
    # This verifies normalization repairs the specific duplicate-ID snapshot pair emitted by some SkillsBench
    # trajectories: the completed result has no call metadata and its following pending snapshot has no result.
    completed = ACPToolCall.model_validate(
        {
            "type": "tool_call",
            "tool_call_id": "shared-id",
            "kind": "tool",
            "title": "",
            "status": "completed",
            "content": [
                {"type": "content", "content": {"type": "text", "text": "Tool: file_editor\nResult:\ncontents"}}
            ],
        }
    )
    pending = ACPToolCall.model_validate(
        {
            "type": "tool_call",
            "tool_call_id": "shared-id",
            "kind": "read",
            "title": "Inspect input: Reading /root/input.txt",
            "status": "pending",
            "content": [],
        }
    )

    events = acp_messages_to_openhands_events([completed, pending])

    actions = [event for event in events if isinstance(event, ActionEvent)]
    assert len(actions) == 1
    assert actions[0].tool_call_id == "shared-id"
    assert actions[0].summary == "Inspect input"
    assert isinstance(actions[0].action, TitleInferredFileEditorAction)
    assert actions[0].action.path == "/root/input.txt"
    assert isinstance(events[1], ObservationEvent)
    assert events[1].observation.text == "contents"


def test_render_acp_trajectory_from_jsonl_path(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "acp_trajectory.jsonl"
    records = [
        ACPUserMessage(type="user_message", text="Run the command."),
        ACPToolCall.model_validate(
            {
                "type": "tool_call",
                "tool_call_id": "call-id",
                "kind": "execute",
                "title": "$ printf ok",
                "status": "completed",
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "Tool: terminal\nResult:\nok"}}
                ],
            }
        ),
    ]
    trajectory_path.write_text("\n".join(record.model_dump_json() for record in records))

    rendered = render_acp_trajectory(trajectory_path)

    assert "<message role=user>" in rendered
    assert "terminal(" in rendered
    assert "printf ok" in rendered
    assert "<observation step=1 tool_call_id=tool_call_0001>" in rendered
    assert "ok" in rendered


def test_acp_normalization_recovers_file_editor_edit_from_title() -> None:
    # This verifies title-only ACP edits retain their recoverable operation and path in both full and condensed
    # OpenHands rendering without claiming the unavailable str-replacement arguments.
    edit = ACPToolCall.model_validate(
        {
            "type": "tool_call",
            "tool_call_id": "edit-id",
            "kind": "edit",
            "title": "Apply fix: Editing /root/input.txt",
            "status": "completed",
            "content": [{"type": "content", "content": {"type": "text", "text": "Tool: file_editor\nResult:\nDone"}}],
        }
    )

    action, observation = acp_messages_to_openhands_events([edit])

    assert isinstance(action, ActionEvent)
    assert isinstance(action.action, ArchivedACPAction)
    assert action.action.arguments == {"path": "/root/input.txt", "operation": "edit"}
    full_rendering = render_trajectory([action, observation])
    assert 'file_editor({"path": "/root/input.txt", "operation": "edit"})' in full_rendering
    condensed = render_trajectory(condense_uncited_action_steps([action, observation], cited_step_numbers=set()))
    assert "file_editor edit: /root/input.txt" in condensed
    assert "Operation: edit\nPath: /root/input.txt\nStatus: success" in condensed


def test_render_trajectory_uses_the_openhands_active_branch(tmp_path: Path) -> None:
    # This verifies archive rendering follows the exact event branch OpenHands exposes to its next LLM call.
    if not ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    archived_state, events = load_conversation_state_and_events_from_archive(ARCHIVE_PATH)
    conversation = LocalConversation(
        agent=archived_state.agent,
        workspace=LocalWorkspace(working_dir=tmp_path),
        persistence_dir=None,
        conversation_id=archived_state.id,
        visualizer=None,
        delete_on_close=False,
    )
    replay_events(conversation, events)

    active_branch = get_active_branch_events(events, archived_state)

    assert active_branch == conversation.state.active_branch()
    assert render_trajectory(events, state=archived_state) == render_trajectory(active_branch)


def test_render_trajectory_can_skip_initial_system_and_user_messages() -> None:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    # This verifies skipped rendering begins at the first action of the active branch.
    state, events = load_conversation_state_and_events_from_archive(ARCHIVE_PATH)
    active_branch = get_active_branch_events(events, state)
    first_action_event_index = next(
        index for index, event in enumerate(active_branch) if isinstance(event, ActionEvent)
    )

    rendered = render_trajectory(events, state=state, start_at_first_action_event=True)

    assert rendered == render_trajectory(active_branch[first_action_event_index:])


def test_render_trajectory_normalizes_tool_call_identifiers_by_default() -> None:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    # This verifies rendering hides original tool identifiers and retains an explicit opt-out for debugging.
    state, events = load_conversation_state_and_events_from_archive(ARCHIVE_PATH)
    first_action = next(event for event in get_active_branch_events(events, state) if isinstance(event, ActionEvent))
    assert first_action.tool_call is not None

    normalized = render_trajectory(events, state=state)
    unnormalized = render_trajectory(events, state=state, normalize_tool_call_ids=False)

    assert "tool_call_0001" in normalized
    assert first_action.tool_call.id not in normalized
    assert first_action.tool_call.id in unnormalized


def test_render_acp_trajectory_matches_astropy_codex_fixture() -> None:
    if not ASTROPY_ACP_ARCHIVE_PATH.is_file() or not ASTROPY_ACP_RENDERED_FIXTURE_PATH.is_file():
        pytest.skip("This test requires locally supplied trajectory and rendering fixtures.")
    # This verifies the complete rendering of a real Codex software-engineering trajectory remains byte-for-byte
    # stable; it assumes the checked-in archive and readable XML fixture are updated together intentionally.
    state, events = load_conversation_state_and_events_from_archive(ASTROPY_ACP_ARCHIVE_PATH)

    rendered = render_acp_trajectory(events, state=state)

    assert rendered == ASTROPY_ACP_RENDERED_FIXTURE_PATH.read_text().removesuffix("\n")


def test_render_acp_trajectory_matches_astropy_claude_code_fixture() -> None:
    if not ASTROPY_CLAUDE_CODE_ARCHIVE_PATH.is_file() or not ASTROPY_CLAUDE_CODE_RENDERED_FIXTURE_PATH.is_file():
        pytest.skip("This test requires locally supplied trajectory and rendering fixtures.")
    # This verifies automatic ACP detection reproduces a complete Claude Code trajectory byte-for-byte and that the
    # shared action-first option removes only its initial messages; it assumes the archive and fixture change together.
    state, events = load_conversation_state_and_events_from_archive(ASTROPY_CLAUDE_CODE_ARCHIVE_PATH)

    rendered = render_acp_trajectory(events, state=state)
    action_first_rendered = render_trajectory(events, state=state, start_at_first_action_event=True)

    assert rendered == ASTROPY_CLAUDE_CODE_RENDERED_FIXTURE_PATH.read_text().removesuffix("\n")
    assert render_trajectory(events, state=state) == rendered
    assert action_first_rendered == rendered[rendered.index("<action step=1>") :]
    assert action_first_rendered.count("<action ") == 20
    assert action_first_rendered.count("<observation ") == 20
    assert "<message role=user>" not in action_first_rendered
    assert "<agent_messages>" not in rendered
    assert rendered.count("<combined_agent_messages>") == 1
    assert "This block contains the combined messages received from the agent throughout the trajectory" in rendered
    assert "finish(" not in rendered


def test_astropy_claude_code_acp_observation_content_schema() -> None:
    if not ASTROPY_CLAUDE_CODE_ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    # This verifies the real Claude Code archive has one content block per terminal observation and distinguishes
    # its nested text blocks from edit diff blocks; it assumes null ACP metadata is preserved during archive loading.
    block_types = _assert_claude_code_acp_observation_content_schema(ASTROPY_CLAUDE_CODE_ARCHIVE_PATH)

    assert block_types == Counter({"content": 18, "diff": 2})


def test_render_acp_trajectory_preserves_interleaved_message_event_position() -> None:
    # This verifies independently archived assistant messages remain at their observed event position rather than
    # being bundled at the end; it assumes event order is the only timing resolution persisted for message text.
    start = ACPToolCallEvent(
        tool_call_id="call",
        title="Run command",
        status="in_progress",
        tool_kind="execute",
        raw_input={"command": "run"},
    )
    message = MessageEvent(
        source="agent",
        llm_message=Message(role="assistant", content=[TextContent(text="Progress update")]),
    )
    terminal = start.model_copy(
        update={"status": "completed", "raw_output": {"formatted_output": "result", "exit_code": 0}}
    )

    rendered = render_acp_trajectory([start, message, terminal])

    assert rendered.index("<action step=1>") < rendered.index("Progress update")
    assert rendered.index("Progress update") < rendered.index("<observation step=1")


def test_render_acp_trajectory_preserves_overlapping_call_order() -> None:
    # This verifies separate action and observation entries expose simultaneous in-flight calls without inventing
    # model-turn boundaries; it assumes archive order is ACP notification order.
    first_start = ACPToolCallEvent(
        tool_call_id="first",
        title="Run first",
        status="in_progress",
        tool_kind="execute",
        raw_input={"command": ["first"]},
    )
    second_start = ACPToolCallEvent(
        tool_call_id="second",
        title="Run second",
        status="in_progress",
        tool_kind="execute",
        raw_input={"command": ["second"]},
    )
    second_terminal = second_start.model_copy(
        update={"status": "completed", "raw_output": {"formatted_output": "second output", "exit_code": 0}}
    )
    first_terminal = first_start.model_copy(
        update={"status": "completed", "raw_output": {"formatted_output": "first output", "exit_code": 0}}
    )

    rendered = render_acp_trajectory([first_start, second_start, second_terminal, first_terminal])

    assert rendered.index("<action step=1") < rendered.index("<action step=2")
    assert rendered.index("<action step=2") < rendered.index("<observation step=2")
    assert rendered.index("<observation step=2") < rendered.index("<observation step=1")


def test_condense_uncited_acp_action_steps_preserves_cited_overlapping_call() -> None:
    # This verifies ACP condensation uses call-order step numbers despite overlapping snapshots, preserves every
    # snapshot of a cited call, and replaces the uncited call with bounded action and head/tail result summaries.
    first_start = ACPToolCallEvent(
        tool_call_id="first",
        title="First " + "input " * 100,
        status="in_progress",
        tool_kind="execute",
        raw_input={"command": "first " * 100},
    )
    second_start = ACPToolCallEvent(
        tool_call_id="second",
        title="Second call",
        status="in_progress",
        tool_kind="execute",
        raw_input={"command": "second"},
    )
    second_terminal = second_start.model_copy(
        update={"status": "completed", "raw_output": {"formatted_output": "second result", "exit_code": 0}}
    )
    first_terminal = first_start.model_copy(
        update={
            "status": "completed",
            "raw_output": {"formatted_output": "head" + "x" * 500 + "tail", "exit_code": 0},
        }
    )

    condensed = condense_uncited_action_steps(
        [first_start, second_start, second_terminal, first_terminal], cited_step_numbers={2}
    )
    rendered = render_trajectory(condensed)

    assert isinstance(condensed[0], SummarizedACPActionEvent)
    assert condensed[1:3] == [second_start, second_terminal]
    assert isinstance(condensed[3], SummarizedACPObservationEvent)
    assert rendered.index("<summarized_action step=1>") < rendered.index("<action step=2>")
    assert rendered.index("<observation step=2") < rendered.index("<summarized_observation step=1>")
    assert "Command: " in rendered
    assert "Title: " not in rendered
    assert "Input: " not in rendered
    assert "..." in rendered
    assert "head" in rendered and "tail" in rendered


def test_condense_uncited_acp_action_prefers_archived_summary_over_command() -> None:
    # This verifies ACP condensation uses the producer's concise semantic description and does not duplicate its
    # potentially large command through title and input fields; it assumes description is producer-authored text.
    start = ACPToolCallEvent(
        tool_call_id="described",
        title="python -c 'large command'",
        status="in_progress",
        tool_kind="execute",
        raw_input={"command": "python -c 'large command'", "description": "Test property inheritance"},
    )
    terminal = start.model_copy(
        update={"status": "completed", "raw_output": {"formatted_output": "passed", "exit_code": 0}}
    )

    rendered = render_trajectory(condense_uncited_action_steps([start, terminal], cited_step_numbers=set()))

    action_summary = rendered.split("</summarized_action>", 1)[0]
    assert action_summary.endswith("Test property inheritance\n")
    assert "Title:" not in action_summary
    assert "Command:" not in action_summary
    assert "Input:" not in action_summary


def test_fully_condensed_acp_trajectory_remains_acp_renderable() -> None:
    # This guarantees summarized ACP subclasses retain ACP routing when no original tool-call snapshots survive;
    # it assumes each call has one start and one terminal snapshot.
    start = ACPToolCallEvent(
        tool_call_id="only",
        title="Only call",
        status="in_progress",
        tool_kind="read",
        raw_input={"path": "/tmp/input"},
    )
    terminal = start.model_copy(update={"status": "failed", "raw_output": "contents", "is_error": True})

    condensed = condense_uncited_action_steps([start, terminal], cited_step_numbers=set())
    rendered = render_trajectory(condensed)

    assert [type(event) for event in condensed] == [SummarizedACPActionEvent, SummarizedACPObservationEvent]
    assert rendered.count("<summarized_action step=1>") == 1
    assert rendered.count("<summarized_observation step=1>") == 1
    assert "read view: /tmp/input" in rendered
    assert "Tool: read\nOperation: view\nPath: /tmp/input\nStatus: failure" in rendered
    assert "Result: contents" in rendered


def test_condense_uncited_acp_file_edit_matches_file_editor_summary_shape() -> None:
    # This verifies an ACP diff observation condenses like the normal file-editor path instead of retaining bounded
    # old/new excerpts; it assumes the diff block supplies path, old_text, and new_text strings.
    start = ACPToolCallEvent(
        tool_call_id="edit",
        title="Edit file",
        status="in_progress",
        tool_kind="edit",
        raw_input={"file_path": "/tmp/a", "old_string": "same\nold", "new_string": "same\nnew"},
    )
    terminal = start.model_copy(
        update={
            "status": "completed",
            "content": [
                {
                    "type": "diff",
                    "field_meta": None,
                    "path": "/tmp/a",
                    "old_text": "same\nold",
                    "new_text": "same\nnew",
                }
            ],
        }
    )

    full_rendering = render_trajectory([start, terminal])
    rendered = render_trajectory(condense_uncited_action_steps([start, terminal], cited_step_numbers=set()))

    assert "edit({" in full_rendering
    assert "Path: /tmp/a\nOld text:\nsame\nold\nNew text:\nsame\nnew" in full_rendering
    assert "edit edit: /tmp/a" in rendered
    assert "Tool: edit\nOperation: edit\nPath: /tmp/a\nStatus: success\nChanged lines: 2" in rendered
    assert "Old text:" not in rendered
    assert "New text:" not in rendered


def test_condense_uncited_acp_file_read_matches_file_editor_summary_shape() -> None:
    # This verifies a full ACP read keeps its recovered file contents while its condensed form uses the normal
    # file-editor view summary; it assumes the native read input identifies the file path.
    start = ACPToolCallEvent(
        tool_call_id="read",
        title="Read file",
        status="in_progress",
        tool_kind="read",
        raw_input={"file_path": "/tmp/a"},
    )
    terminal = start.model_copy(
        update={
            "status": "completed",
            "content": [
                {
                    "type": "content",
                    "field_meta": None,
                    "content": {"type": "text", "text": "line one  \nline two"},
                }
            ],
        }
    )

    full_rendering = render_trajectory([start, terminal])
    condensed = render_trajectory(condense_uncited_action_steps([start, terminal], cited_step_numbers=set()))

    assert "read({" in full_rendering
    assert "line one\nline two" in full_rendering
    assert "read view: /tmp/a" in condensed
    assert "Tool: read\nOperation: view\nPath: /tmp/a\nStatus: success" in condensed
    assert "line one" not in condensed


def test_condense_uncited_acp_pathless_read_falls_back_to_command_summary() -> None:
    # This verifies a Codex unified-exec call classified as read can condense as a command when it has no singular
    # file path, while the file-operation path remains strict for malformed records without a command.
    start = ACPToolCallEvent(
        tool_call_id="command-read",
        title="Read a.py, Read b.py",
        status="in_progress",
        tool_kind="read",
        raw_input={
            "command": ["/bin/bash", "-lc", "sed -n '1,20p' a.py; sed -n '1,20p' b.py"],
            "cwd": "/workspace",
        },
    )
    terminal = start.model_copy(
        update={
            "status": "completed",
            "raw_output": {"formatted_output": "a contents\nb contents", "exit_code": 0},
        }
    )

    rendered = render_trajectory(condense_uncited_action_steps([start, terminal], cited_step_numbers=set()))

    assert "Tool: read\nCommand:" in rendered
    assert "sed -n '1,20p' a.py" in rendered
    assert "Tool: read\nStatus: completed\nOutput:\na contents\nb contents" in rendered
    assert "Operation: view" not in rendered


def test_condense_uncited_acp_failed_pathless_read_falls_back_to_failure_summary() -> None:
    # This verifies a definitively failed Claude read rejected before its arguments were archived remains renderable
    # from its title and error, while the fallback is limited to the terminal failed status.
    start = ACPToolCallEvent(
        tool_call_id="failed-read",
        title="Read File",
        status="pending",
        tool_kind="read",
        raw_input={},
    )
    terminal = start.model_copy(
        update={
            "status": "failed",
            "raw_output": "InputValidationError: offset must be a number",
            "content": [
                {
                    "type": "content",
                    "content": {"type": "text", "text": "InputValidationError: offset must be a number"},
                }
            ],
        }
    )

    rendered = render_trajectory(condense_uncited_action_steps([start, terminal], cited_step_numbers=set()))

    assert "Tool: read\nTitle: Read File" in rendered
    assert "Tool: read\nStatus: failed\nOutput:\nInputValidationError: offset must be a number" in rendered


def test_render_and_condense_acp_file_creation_without_old_text() -> None:
    # This verifies Claude write/create diffs accept null old_text, omit a fictitious old-file section in full
    # rendering, and retain the new file's changed-line range without inferring a more specific edit operation.
    start = ACPToolCallEvent(
        tool_call_id="create",
        title="Write new.py",
        status="in_progress",
        tool_kind="edit",
        raw_input={"file_path": "/tmp/new.py", "content": "first\nsecond"},
    )
    terminal = start.model_copy(
        update={
            "status": "completed",
            "content": [
                {
                    "type": "diff",
                    "path": "/tmp/new.py",
                    "old_text": None,
                    "new_text": "first\nsecond",
                }
            ],
        }
    )

    full_rendering = render_trajectory([start, terminal])
    condensed = render_trajectory(condense_uncited_action_steps([start, terminal], cited_step_numbers=set()))

    assert "Path: /tmp/new.py\nNew text:\nfirst\nsecond" in full_rendering
    assert "Old text:" not in full_rendering
    assert "edit edit: /tmp/new.py" in condensed
    assert "Tool: edit\nOperation: edit\nPath: /tmp/new.py\nStatus: success\nChanged lines: 1-2" in condensed


def test_render_acp_trajectory_rejects_missing_command_formatted_output() -> None:
    # This verifies a command terminal event cannot silently fall back to aggregated output when the model-visible
    # formatted output is absent; it assumes command results identify themselves through their output schema.
    start = ACPToolCallEvent(
        tool_call_id="command",
        title="Run command",
        status="in_progress",
        tool_kind="execute",
    )
    terminal = start.model_copy(
        update={"status": "completed", "raw_output": {"aggregated_output": "raw output", "exit_code": 0}}
    )

    with pytest.raises(ValueError, match="missing formatted_output"):
        render_acp_trajectory([start, terminal])


def test_render_acp_trajectory_extracts_known_content_blocks_and_preserves_unknown_blocks() -> None:
    # This verifies nested ACP text and file-editor diff blocks render their meaningful payloads while an unknown
    # block remains losslessly JSON-serialized; it assumes raw_output is absent so content has precedence.
    cases = [
        (
            "text",
            [{"type": "content", "field_meta": None, "content": {"type": "text", "text": "output"}}],
            "output",
            '"field_meta"',
        ),
        (
            "diff",
            [{"type": "diff", "field_meta": None, "path": "/tmp/a", "old_text": "old", "new_text": "new"}],
            "Path: /tmp/a\nOld text:\nold\nNew text:\nnew",
            '"field_meta"',
        ),
        (
            "unknown",
            [{"type": "content", "field_meta": None, "content": {"type": "resource_link", "uri": "x"}}],
            '"type": "resource_link"',
            None,
        ),
    ]

    for tool_call_id, content, expected, absent in cases:
        start = ACPToolCallEvent(
            tool_call_id=tool_call_id,
            title=tool_call_id,
            status="in_progress",
            tool_kind="read",
        )
        terminal = start.model_copy(update={"status": "completed", "content": content})

        rendered = render_acp_trajectory([start, terminal])

        assert expected in rendered
        if absent is not None:
            assert absent not in rendered


def test_render_acp_trajectory_truncates_each_formatted_output() -> None:
    # This verifies ACP observations retain the existing per-event character ceiling and an explicit truncation
    # marker; it assumes the terminal formatted output is otherwise valid.
    start = ACPToolCallEvent(
        tool_call_id="long-command",
        title="Run long command",
        status="in_progress",
        tool_kind="execute",
    )
    terminal = start.model_copy(
        update={"status": "completed", "raw_output": {"formatted_output": "x" * 60_000, "exit_code": 0}}
    )

    rendered = render_acp_trajectory([start, terminal])

    assert TRUNCATED_CONTENT_NOTE in rendered
    assert "x" * 50_000 not in rendered
