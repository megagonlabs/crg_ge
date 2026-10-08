import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from openhands.sdk import Message, TextContent
from openhands.sdk.conversation import ConversationState
from openhands.sdk.event import (
    ACPToolCallEvent,
    ActionEvent,
    ConversationStateUpdateEvent,
    Event,
    LLMConvertibleEvent,
    MessageEvent,
    ObservationEvent,
    SystemPromptEvent,
)
from openhands.sdk.event.condenser import CondensationSummaryEvent
from openhands.sdk.event.llm_convertible.observation import ObservationBaseEvent
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.tool import Action, Observation
from openhands.sdk.tool.builtins.finish import FinishAction
from openhands.sdk.tool.builtins.think import ThinkAction, ThinkObservation
from openhands.tools.file_editor.definition import FileEditorAction
from openhands.tools.terminal.definition import TerminalAction
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_core import to_jsonable_python

from crg_ce.graph.graph_generators.gsn.event_summarizers import (
    summarize_acp_action,
    summarize_acp_observation,
    summarize_action,
    summarize_observation,
)
from crg_ce.utils.openhands import index_actions

MAX_RENDERED_EVENT_CONTENT_CHARS = 50_000
TRUNCATED_CONTENT_NOTE = "\n\n[Output truncated for length.]"
ACP_WEB_OUTPUT_NOTE = "Web-search results were not separately archived; subsequent agent text may reflect them."
_ACP_TOOL_RESULT_HEADER = re.compile(r"^Tool: (?P<name>[^\n]+)\nResult:\n")
_ACP_INVOKED_SKILL = re.compile(r"^\[skill: (?P<name>[^\]]+)]")
_ACP_CRITIC_SUFFIX = re.compile(r"^Critic:.*", re.MULTILINE | re.DOTALL)


class _ACPBaseModel(BaseModel):
    """Base model that preserves archive fields added by ACP producers."""

    model_config = ConfigDict(extra="allow")


class ACPTextContent(_ACPBaseModel):
    text: str
    type: Literal["text"]


class ACPContentBlock(_ACPBaseModel):
    content: ACPTextContent
    type: Literal["content"]


class ACPUserMessage(_ACPBaseModel):
    type: Literal["user_message"]
    text: str


class ACPAgentThought(_ACPBaseModel):
    type: Literal["agent_thought"]
    text: str


class ACPAgentMessage(_ACPBaseModel):
    type: Literal["agent_message"]
    text: str


class ACPToolCall(_ACPBaseModel):
    type: Literal["tool_call"]
    tool_call_id: str
    kind: str
    title: str
    status: str
    content: list[ACPContentBlock]


class ArchivedACPAction(Action):
    """OpenHands action retaining ACP fields that cannot form a concrete action."""

    arguments: dict[str, Any]
    arguments_complete: bool
    title: str
    tool_kind: str


class ArchivedACPObservation(Observation):
    """OpenHands observation retaining the original ACP result representation."""

    status: str
    archived_content: list[dict[str, Any]]


class TitleInferredTerminalAction(TerminalAction):
    """Terminal action whose command was recovered from a lossy ACP title."""

    arguments_complete: Literal[False] = Field(default=False, exclude=True)


class TitleInferredFileEditorAction(FileEditorAction):
    """File-editor action whose arguments were recovered from a lossy ACP title."""

    arguments_complete: Literal[False] = Field(default=False, exclude=True)


type ACPMessage = Annotated[
    ACPUserMessage | ACPAgentThought | ACPAgentMessage | ACPToolCall,
    Field(discriminator="type"),
]
_ACP_MESSAGE_ADAPTER: TypeAdapter[ACPMessage] = TypeAdapter(ACPMessage)


def load_acp_jsonl_trajectory(trajectory_path: str | Path) -> list[ACPMessage]:
    """Deserialize an ACP JSONL archive without discarding extension fields."""
    with Path(trajectory_path).open() as trajectory:
        return [_ACP_MESSAGE_ADAPTER.validate_json(line) for line in trajectory]


def _parse_acp_tool_call(message: ACPToolCall) -> tuple[str, dict[str, Any], str, bool]:
    """Recover the tool name, arguments, summary, and completeness from an ACP record."""
    output = "\n".join(content_block.content.text for content_block in message.content)
    result_header = _ACP_TOOL_RESULT_HEADER.match(output)
    tool_name = (
        result_header["name"]
        if result_header is not None
        else {
            "execute": "terminal",
            "read": "file_editor",
            "edit": "file_editor",
            "skill": "invoke_skill",
        }.get(message.kind, "unknown_acp_tool")
    )

    if tool_name == "invoke_skill" and result_header is not None:
        invoked_skill = _ACP_INVOKED_SKILL.match(output[result_header.end() :])
        if invoked_skill is not None:
            return tool_name, {"name": invoked_skill["name"]}, message.title, True

    if tool_name == "terminal":
        if ": $ " in message.title:
            summary, command = message.title.split(": $ ", 1)
            return tool_name, {"command": command}, summary, False
        if message.title.startswith("$ "):
            return tool_name, {"command": message.title[2:]}, "", False

    if tool_name == "file_editor":
        file_editor_operations = (("Reading", "view"), ("Editing", "edit"))
        for operation, file_editor_command in file_editor_operations:
            marker = f": {operation} "
            if marker in message.title:
                summary, path = message.title.split(marker, 1)
                arguments = {"path": path}
                if file_editor_command == "view":
                    arguments["command"] = file_editor_command
                elif file_editor_command is not None:
                    arguments["operation"] = file_editor_command
                return tool_name, arguments, summary, False
            prefix = f"{operation} "
            if message.title.startswith(prefix):
                arguments = {"path": message.title.removeprefix(prefix)}
                if file_editor_command == "view":
                    arguments["command"] = file_editor_command
                elif file_editor_command is not None:
                    arguments["operation"] = file_editor_command
                return tool_name, arguments, "", False

    return tool_name, {}, message.title, False


def _coalesce_acp_tool_call_snapshots(messages: Sequence[ACPMessage]) -> list[ACPMessage]:
    """Merge the malformed completed-then-pending snapshots emitted for one ACP tool call."""
    coalesced: list[ACPMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        following = messages[index + 1] if index + 1 < len(messages) else None
        if (
            isinstance(message, ACPToolCall)
            and isinstance(following, ACPToolCall)
            and (message.tool_call_id == following.tool_call_id)
        ):
            if not (
                message.status in {"completed", "failed"}
                and message.kind in {"tool", "skill"}
                and not message.title
                and message.content
                and following.status == "pending"
                and following.kind in {"read", "edit", "other"}
                and following.title
                and not following.content
            ):
                raise ValueError(f"Unexpected duplicate ACP tool-call id: {message.tool_call_id}")
            coalesced.append(message.model_copy(update={"kind": following.kind, "title": following.title}))
            index += 2
            continue
        coalesced.append(message)
        index += 1
    return coalesced


def acp_messages_to_openhands_events(messages: Sequence[ACPMessage]) -> list[Event]:
    """Convert combined ACP JSONL records into renderable OpenHands events."""
    events: list[Event] = []
    synthetic_thought_index = 0

    for record_index, message in enumerate(_coalesce_acp_tool_call_snapshots(messages), start=1):
        parent_id = events[-1].id if events else None
        if isinstance(message, ACPUserMessage):
            events.append(
                MessageEvent(
                    source="user",
                    parent_id=parent_id,
                    llm_message=Message(role="user", content=[TextContent(text=message.text)]),
                )
            )
            continue

        if isinstance(message, ACPAgentMessage):
            tool_call_id = f"acp-finish-{record_index:04d}"
            finish_message = _ACP_CRITIC_SUFFIX.sub("", message.text).rstrip()
            events.append(
                ActionEvent(
                    parent_id=parent_id,
                    action=FinishAction(message=finish_message),
                    thought=[],
                    tool_name="finish",
                    tool_call_id=tool_call_id,
                    tool_call=MessageToolCall(
                        id=tool_call_id,
                        name="finish",
                        arguments=json.dumps({"message": finish_message}, ensure_ascii=False),
                        origin="completion",
                    ),
                    llm_response_id=f"acp-jsonl-{record_index}",
                )
            )
            continue

        if isinstance(message, ACPAgentThought):
            synthetic_thought_index += 1
            tool_call_id = f"acp-think-{synthetic_thought_index:04d}"
            thought_action = ThinkAction(thought=message.text)
            action_event = ActionEvent(
                parent_id=parent_id,
                action=thought_action,
                thought=[],
                tool_name="think",
                tool_call_id=tool_call_id,
                tool_call=MessageToolCall(
                    id=tool_call_id,
                    name="think",
                    arguments=json.dumps({"thought": message.text}, ensure_ascii=False),
                    origin="completion",
                ),
                llm_response_id=f"acp-jsonl-{record_index}",
            )
            events.append(action_event)
            events.append(
                ObservationEvent(
                    parent_id=action_event.id,
                    observation=ThinkObservation.from_text("Your thought has been logged."),
                    tool_name="think",
                    tool_call_id=tool_call_id,
                    action_id=action_event.id,
                )
            )
            continue

        if not isinstance(message, ACPToolCall):
            raise ValueError(f"Unsupported ACP record type: {message.type!r}")

        tool_name, arguments, summary, arguments_complete = _parse_acp_tool_call(message)
        if tool_name == "terminal" and set(arguments) == {"command"}:
            tool_action: Action = TitleInferredTerminalAction(command=arguments["command"])
        elif tool_name == "file_editor" and set(arguments) == {"command", "path"}:
            tool_action = TitleInferredFileEditorAction(command=arguments["command"], path=arguments["path"])
        else:
            tool_action = ArchivedACPAction(
                arguments=arguments,
                arguments_complete=arguments_complete,
                title=message.title,
                tool_kind=message.kind,
            )

        action_event = ActionEvent(
            parent_id=parent_id,
            action=tool_action,
            thought=[],
            tool_name=tool_name,
            tool_call_id=message.tool_call_id,
            tool_call=MessageToolCall(
                id=message.tool_call_id,
                name=tool_name,
                arguments=json.dumps(arguments, ensure_ascii=False),
                origin="completion",
            ),
            llm_response_id=f"acp-jsonl-{record_index}",
            summary=summary or None,
        )
        events.append(action_event)

        output = "\n".join(content_block.content.text for content_block in message.content)
        result_header = _ACP_TOOL_RESULT_HEADER.match(output)
        result_text = output[result_header.end() :] if result_header is not None else output
        events.append(
            ObservationEvent(
                parent_id=action_event.id,
                observation=ArchivedACPObservation.from_text(
                    result_text,
                    is_error=message.status == "failed",
                    status=message.status,
                    archived_content=[content.model_dump(mode="json") for content in message.content],
                ),
                tool_name=tool_name,
                tool_call_id=message.tool_call_id,
                action_id=action_event.id,
            )
        )

    return events


@dataclass(frozen=True)
class _ACPToolCall:
    step: int
    tool_call_id: str
    first_event_index: int
    terminal_event_index: int
    title: str
    tool_kind: str | None
    raw_input: Any | None
    terminal_event: ACPToolCallEvent


class SummarizedACPActionEvent(ACPToolCallEvent):
    """Synthetic ACP event containing a bounded summary of one tool invocation."""

    step: int
    condensed_summary: str


class SummarizedACPObservationEvent(ACPToolCallEvent):
    """Synthetic ACP event containing a bounded summary of one tool result."""

    step: int
    condensed_summary: str


def trajectory_events_to_messages(events: Sequence[Event]) -> list[Message]:
    llm_events = [event for event in events if isinstance(event, LLMConvertibleEvent)]
    return LLMConvertibleEvent.events_to_messages(llm_events)


def remap_tool_call_ids(messages: Sequence[Message]) -> tuple[list[Message], dict[str, str]]:
    """Replace tool-call identifiers with short stable identifiers in their message order."""
    ids: dict[str, str] = {}

    def remapped_id(tool_call_id: str) -> str:
        if tool_call_id not in ids:
            ids[tool_call_id] = f"tool_call_{len(ids) + 1:04d}"
        return ids[tool_call_id]

    remapped_messages: list[Message] = []
    for message in messages:
        tool_calls = [
            tool_call.model_copy(update={"id": remapped_id(tool_call.id)}) for tool_call in message.tool_calls or []
        ]
        remapped_messages.append(
            message.model_copy(
                update={
                    "tool_calls": tool_calls or None,
                    "tool_call_id": (remapped_id(message.tool_call_id) if message.tool_call_id is not None else None),
                }
            )
        )
    return remapped_messages, ids


def get_active_branch_events(events: Sequence[Event], state: ConversationState | None) -> list[Event]:
    """Return the event branch OpenHands exposes to its agent, including for archived state."""
    if state is None:
        return list(events)

    try:
        return state.active_branch()
    except AttributeError:
        pass

    if state.head_is_empty:
        return []
    leaf_event_id = state.leaf_event_id
    if leaf_event_id is None:
        if not events or events[-1].parent_id is not None:
            return []
        leaf_event_id = events[-1].id

    events_by_id = {event.id: event for event in events}
    if len(events_by_id) != len(events):
        raise ValueError("Cannot reconstruct active branch from duplicate event ids")

    branch_reversed: list[Event] = []
    visited_event_ids: set[str] = set()
    while leaf_event_id is not None:
        if leaf_event_id in visited_event_ids:
            raise ValueError(f"Cycle in active event branch at {leaf_event_id}")
        visited_event_ids.add(leaf_event_id)
        event = events_by_id.get(leaf_event_id)
        if event is None:
            raise ValueError(f"Active branch references unknown event {leaf_event_id}")
        branch_reversed.append(event)
        leaf_event_id = event.parent_id
    return list(reversed(branch_reversed))


def condense_uncited_action_steps(events: Sequence[Event], cited_step_numbers: set[int]) -> list[Event]:
    """Preserve cited action steps verbatim and replace all other actions and observations with summaries."""
    if any(isinstance(event, ACPToolCallEvent) for event in events):
        return _condense_uncited_acp_action_steps(events, cited_step_numbers)

    actions_by_id = index_actions(events)
    action_metadata = {action.id: (step_number, action) for step_number, action, _ in actions_by_id.values()}
    action_metadata_by_tool_call_id = {
        action.tool_call_id: (step_number, action) for step_number, action in action_metadata.values()
    }

    condensed_events: list[Event] = []
    for event in events:
        if isinstance(event, ActionEvent):
            step_number, action = action_metadata[event.id]
            if step_number in cited_step_numbers:
                condensed_events.append(event)
                continue
            try:
                action_summary = summarize_action(action)
            except ValueError:
                raise ValueError(
                    f"Uncited action event {action.id} for tool {action.tool_name} "
                    f"at step {step_number} is missing a summary"
                ) from None
            condensed_events.append(
                CondensationSummaryEvent(
                    id=event.id,
                    timestamp=event.timestamp,
                    parent_id=event.parent_id,
                    summary=(f"<summarized_action step={step_number}>\n{action_summary}\n</summarized_action>"),
                )
            )
            continue

        if isinstance(event, ObservationBaseEvent):
            action_metadata_entry = action_metadata_by_tool_call_id.get(event.tool_call_id)
            if action_metadata_entry is None:
                raise ValueError(f"Observation event {event.id} references unknown tool call id: {event.tool_call_id}")
            step_number, action = action_metadata_entry
            if step_number in cited_step_numbers:
                condensed_events.append(event)
            else:
                observation_summary = summarize_observation(action, event)
                condensed_events.append(
                    CondensationSummaryEvent(
                        id=event.id,
                        timestamp=event.timestamp,
                        parent_id=event.parent_id,
                        summary=(
                            f"<summarized_observation step={step_number}>\n"
                            f"{observation_summary}\n"
                            "</summarized_observation>"
                        ),
                    )
                )
            continue

        condensed_events.append(event)

    return condensed_events


def _truncate_rendered_event_content(content: str) -> str:
    if len(content) <= MAX_RENDERED_EVENT_CONTENT_CHARS:
        return content
    return content[: MAX_RENDERED_EVENT_CONTENT_CHARS - len(TRUNCATED_CONTENT_NOTE)].rstrip() + TRUNCATED_CONTENT_NOTE


def _render_acp_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(to_jsonable_python(value), indent=2, ensure_ascii=False, sort_keys=True)


def _index_acp_tool_calls(events: Sequence[Event]) -> tuple[dict[int, _ACPToolCall], dict[int, _ACPToolCall]]:
    snapshots_by_tool_call_id: dict[str, list[tuple[int, ACPToolCallEvent]]] = {}
    steps_by_tool_call_id: dict[str, int] = {}
    for event_index, event in enumerate(events):
        if not isinstance(event, ACPToolCallEvent):
            continue
        if event.tool_call_id not in steps_by_tool_call_id:
            steps_by_tool_call_id[event.tool_call_id] = len(steps_by_tool_call_id) + 1
        if isinstance(event, SummarizedACPActionEvent | SummarizedACPObservationEvent):
            if event.step != steps_by_tool_call_id[event.tool_call_id]:
                raise ValueError(
                    f"Summarized ACP tool call {event.tool_call_id} has step {event.step}, "
                    f"expected {steps_by_tool_call_id[event.tool_call_id]}"
                )
        else:
            snapshots_by_tool_call_id.setdefault(event.tool_call_id, []).append((event_index, event))

    calls_by_first_event_index: dict[int, _ACPToolCall] = {}
    calls_by_terminal_event_index: dict[int, _ACPToolCall] = {}
    for tool_call_id, snapshots in snapshots_by_tool_call_id.items():
        step = steps_by_tool_call_id[tool_call_id]
        terminal_snapshots = [
            (event_index, event) for event_index, event in snapshots if event.status in {"completed", "failed"}
        ]
        if len(terminal_snapshots) != 1:
            raise ValueError(
                f"Expected exactly one terminal ACP event for tool call {tool_call_id}, found {len(terminal_snapshots)}"
            )
        terminal_event_index, terminal_event = terminal_snapshots[0]
        if snapshots[-1][0] != terminal_event_index:
            raise ValueError(f"ACP tool call {tool_call_id} contains snapshots after its terminal event")

        title = terminal_event.title
        tool_kind = terminal_event.tool_kind
        raw_input = terminal_event.raw_input
        for _, snapshot in reversed(snapshots):
            if not title and snapshot.title:
                title = snapshot.title
            if tool_kind is None and snapshot.tool_kind is not None:
                tool_kind = snapshot.tool_kind
            if raw_input is None and snapshot.raw_input is not None:
                raw_input = snapshot.raw_input

        call = _ACPToolCall(
            step=step,
            tool_call_id=tool_call_id,
            first_event_index=snapshots[0][0],
            terminal_event_index=terminal_event_index,
            title=title,
            tool_kind=tool_kind,
            raw_input=raw_input,
            terminal_event=terminal_event,
        )
        calls_by_first_event_index[call.first_event_index] = call
        calls_by_terminal_event_index[call.terminal_event_index] = call
    return calls_by_first_event_index, calls_by_terminal_event_index


def _condense_uncited_acp_action_steps(events: Sequence[Event], cited_step_numbers: set[int]) -> list[Event]:
    calls_by_start, calls_by_terminal = _index_acp_tool_calls(events)
    calls_by_tool_call_id = {call.tool_call_id: call for call in calls_by_start.values()}
    condensed_events: list[Event] = []
    for event_index, event in enumerate(events):
        if isinstance(event, SummarizedACPActionEvent | SummarizedACPObservationEvent):
            condensed_events.append(event)
            continue
        if not isinstance(event, ACPToolCallEvent):
            condensed_events.append(event)
            continue

        start_call = calls_by_start.get(event_index)
        terminal_call = calls_by_terminal.get(event_index)
        call = calls_by_tool_call_id.get(event.tool_call_id)
        if call is None:
            raise ValueError(f"ACP event {event.id} references an unindexed tool call: {event.tool_call_id}")
        if call.step in cited_step_numbers:
            condensed_events.append(event)
            continue
        file_editor_operation, file_editor_path = _acp_file_editor_operation(call)
        if start_call is not None:
            raw_input = start_call.raw_input if isinstance(start_call.raw_input, dict) else {}
            action_summary = raw_input.get("description") or raw_input.get("summary")
            if not isinstance(action_summary, str):
                action_summary = None
            command = raw_input.get("command")
            condensed_events.append(
                SummarizedACPActionEvent.model_validate(
                    {
                        **event.model_dump(mode="python", exclude={"id"}, exclude_computed_fields=True),
                        "step": start_call.step,
                        "condensed_summary": summarize_acp_action(
                            start_call.tool_kind or "unknown",
                            start_call.title,
                            _render_acp_value(_acp_action_input_value(start_call)),
                            action_summary=action_summary,
                            rendered_command=_render_acp_value(command) if command is not None else None,
                            file_editor_operation=file_editor_operation,
                            file_editor_path=file_editor_path,
                        ),
                    }
                )
            )
        if terminal_call is not None:
            rendered_output, _ = _acp_rendered_output(terminal_call)
            condensed_events.append(
                SummarizedACPObservationEvent.model_validate(
                    {
                        **event.model_dump(mode="python", exclude={"id"}, exclude_computed_fields=True),
                        "step": terminal_call.step,
                        "condensed_summary": summarize_acp_observation(
                            terminal_call.tool_kind or "unknown",
                            terminal_call.terminal_event.status,
                            rendered_output,
                            file_edits=_extract_acp_file_edits(terminal_call.terminal_event.content or []),
                            file_editor_operation=file_editor_operation,
                            file_editor_path=file_editor_path,
                        ),
                    }
                )
            )
    return condensed_events


def _acp_file_editor_operation(call: _ACPToolCall) -> tuple[str | None, str | None]:
    if call.tool_kind not in {"read", "edit"}:
        return None, None

    operation = "view" if call.tool_kind == "read" else "edit"
    if isinstance(call.raw_input, dict):
        path = call.raw_input.get("file_path", call.raw_input.get("path"))
        if isinstance(path, str):
            return operation, path

        changes = call.raw_input.get("changes")
        if isinstance(changes, dict) and len(changes) == 1:
            path = next(iter(changes))
            if isinstance(path, str):
                return operation, path

    file_edits = _extract_acp_file_edits(call.terminal_event.content or [])
    if file_edits is not None and len(file_edits) == 1:
        return operation, file_edits[0][0]
    if isinstance(call.raw_input, dict) and "command" in call.raw_input:
        return None, None
    if call.terminal_event.status == "failed":
        return None, None
    raise ValueError(f"ACP {call.tool_kind} call {call.tool_call_id} has no identifiable file path")


def _acp_action_input_value(call: _ACPToolCall) -> dict[str, Any]:
    input_value = call.raw_input
    if isinstance(input_value, dict):
        input_value = {
            key: value
            for key, value in input_value.items()
            if key not in {"call_id", "process_id", "turn_id", "parsed_cmd", "source"}
        }
    if input_value is None:
        input_value = {}
    if not isinstance(input_value, dict):
        input_value = {"input": input_value}
    input_value["summary"] = input_value.pop("description", call.title)
    return input_value


def _render_acp_action(call: _ACPToolCall, rendered_tool_call_id: str) -> str:
    rendered_input = _render_acp_value(_acp_action_input_value(call))
    return (
        f"<action step={call.step}>\n"
        "<content>\n<empty>\n</content>\n"
        "<tool_calls>\n"
        f"{_render_tool_call(call.tool_kind or 'unknown', rendered_input, rendered_tool_call_id)}\n"
        "</tool_calls>\n"
        "</action>"
    )


def _extract_acp_file_edits(content: list[dict[str, Any]]) -> list[tuple[str, str | None, str]] | None:
    if not content:
        return None
    file_edits: list[tuple[str, str | None, str]] = []
    for block in content:
        if block.get("type") != "diff":
            return None
        path = block.get("path")
        old_text = block.get("old_text")
        new_text = block.get("new_text")
        if not isinstance(path, str) or not isinstance(new_text, str) or not isinstance(old_text, str | None):
            return None
        file_edits.append((path, old_text, new_text))
    return file_edits


def _extract_recognized_acp_content(content: list[dict[str, Any]]) -> str | None:
    text_blocks: list[str] = []
    for block in content:
        inner = block.get("content")
        if block.get("type") != "content" or not isinstance(inner, dict):
            break
        text = inner.get("text")
        if inner.get("type") != "text" or not isinstance(text, str):
            break
        text_blocks.append("\n".join(line.rstrip() for line in text.split("\n")))
    else:
        return "\n".join(text_blocks)

    file_edits = _extract_acp_file_edits(content)
    if file_edits is not None:
        return "\n\n".join(
            (
                f"Path: {path}\nNew text:\n{new_text}"
                if old_text is None
                else f"Path: {path}\nOld text:\n{old_text}\nNew text:\n{new_text}"
            )
            for path, old_text, new_text in file_edits
        )

    return None


def _acp_rendered_output(call: _ACPToolCall) -> tuple[str, object | None]:
    terminal_event = call.terminal_event
    raw_output = terminal_event.raw_output
    exit_code: object | None = None
    if isinstance(raw_output, dict) and (
        "command" in raw_output or "aggregated_output" in raw_output or "formatted_output" in raw_output
    ):
        if "formatted_output" not in raw_output or not isinstance(raw_output["formatted_output"], str):
            raise ValueError(f"Command ACP tool call {call.tool_call_id} is missing formatted_output")
        rendered_output = raw_output["formatted_output"] or "<empty>"
        exit_code = raw_output.get("exit_code")
    elif terminal_event.content:
        extracted_content = _extract_recognized_acp_content(terminal_event.content)
        rendered_output = (
            extracted_content if extracted_content is not None else _render_acp_value(terminal_event.content)
        )
    elif raw_output is not None:
        rendered_output = _render_acp_value(raw_output)
    elif call.tool_kind == "fetch":
        rendered_output = ACP_WEB_OUTPUT_NOTE
    else:
        raise ValueError(f"ACP tool call {call.tool_call_id} has no archived output")
    return rendered_output, exit_code


def _render_acp_observation(call: _ACPToolCall, rendered_tool_call_id: str) -> str:
    terminal_event = call.terminal_event
    rendered_output, exit_code = _acp_rendered_output(call)

    exit_code_line = "" if exit_code is None else f"<exit_code>{exit_code}</exit_code>\n"
    return (
        f"<observation step={call.step} tool_call_id={rendered_tool_call_id}"
        f" status={terminal_event.status}>\n"
        f"{exit_code_line}"
        f"<content>\n{_truncate_rendered_event_content(rendered_output)}\n</content>\n"
        "</observation>"
    )


def render_acp_trajectory(
    events: Sequence[Event] | str | Path,
    *,
    state: ConversationState | None = None,
    start_at_first_action_event: bool = False,
    normalize_tool_call_ids: bool = True,
) -> str:
    """Render either an ACP event sequence or a SkillsBench ACP JSONL archive."""
    if isinstance(events, str | Path):
        if state is not None:
            raise ValueError("An ACP JSONL archive path cannot be rendered with ConversationState")
        messages = load_acp_jsonl_trajectory(events)
        if start_at_first_action_event:
            messages = [
                message
                for message in messages
                if not (isinstance(message, ACPAgentThought) and message.text.startswith("System Prompt:"))
            ]
        return render_trajectory(
            acp_messages_to_openhands_events(messages),
            start_at_first_action_event=start_at_first_action_event,
        )

    active_events = get_active_branch_events(events, state)
    if not any(isinstance(event, ACPToolCallEvent) for event in active_events):
        raise ValueError("Cannot render an ACP trajectory without ACPToolCallEvent entries")
    if start_at_first_action_event:
        first_action_event_index = next(
            (index for index, event in enumerate(active_events) if isinstance(event, ACPToolCallEvent | ActionEvent)),
            None,
        )
        if first_action_event_index is None:
            raise ValueError("Cannot skip system and user messages from an ACP trajectory without an action event")
        active_events = active_events[first_action_event_index:]
    calls_by_start, calls_by_terminal = _index_acp_tool_calls(active_events)
    rendered_ids = {
        call.tool_call_id: (f"tool_call_{call.step:04d}" if normalize_tool_call_ids else call.tool_call_id)
        for call in calls_by_start.values()
    }
    finish_actions: list[tuple[int, ActionEvent, FinishAction]] = []
    for event_index, candidate_event in enumerate(active_events):
        if isinstance(candidate_event, ActionEvent) and isinstance(candidate_event.action, FinishAction):
            finish_actions.append((event_index, candidate_event, candidate_event.action))
    finish_tool_call_ids = {event.tool_call_id for _, event, _ in finish_actions}
    duplicate_finish_message_indices: set[int] = set()
    for event_index, _, finish_action in finish_actions:
        if event_index == 0:
            continue
        preceding_message = active_events[event_index - 1]
        if not isinstance(preceding_message, MessageEvent):
            continue
        message = preceding_message.to_llm_message()
        content = "\n".join(item.text for item in message.content if isinstance(item, TextContent)).strip()
        if message.role == "assistant" and content == finish_action.message:
            duplicate_finish_message_indices.add(event_index - 1)

    rendered_events: list[str] = []
    for event_index, event in enumerate(active_events):
        if isinstance(event, SummarizedACPActionEvent):
            rendered_events.append(
                f"<summarized_action step={event.step}>\n{event.condensed_summary}\n</summarized_action>"
            )
            continue
        if isinstance(event, SummarizedACPObservationEvent):
            rendered_events.append(
                f"<summarized_observation step={event.step}>\n{event.condensed_summary}\n</summarized_observation>"
            )
            continue

        call = calls_by_start.get(event_index)
        if call is not None:
            rendered_events.append(_render_acp_action(call, rendered_ids[call.tool_call_id]))

        terminal_call = calls_by_terminal.get(event_index)
        if terminal_call is not None:
            rendered_events.append(_render_acp_observation(terminal_call, rendered_ids[terminal_call.tool_call_id]))

        if isinstance(event, ACPToolCallEvent | SystemPromptEvent | ConversationStateUpdateEvent):
            continue
        if isinstance(event, MessageEvent):
            message = event.to_llm_message()
            content = "\n".join(item.text for item in message.content if isinstance(item, TextContent)).strip()
            if event_index in duplicate_finish_message_indices:
                continue
            rendered_events.append(
                f"<message role={message.role}>\n"
                f"<content>\n{_truncate_rendered_event_content(content or '<empty>')}\n</content>\n"
                "</message>"
            )
            continue
        if isinstance(event, ActionEvent):
            if not isinstance(event.action, FinishAction):
                raise ValueError(f"Unsupported non-finish ActionEvent in ACP trajectory: {event.id}")
            rendered_events.append(
                "<combined_agent_messages>\n"
                "<context>This block contains the combined messages received from the agent throughout the "
                "trajectory, including any final message</context>\n"
                f"<content>\n{_truncate_rendered_event_content(event.action.message or '<empty>')}\n</content>\n"
                "</combined_agent_messages>"
            )
            continue
        if isinstance(event, ObservationBaseEvent) and event.tool_call_id in finish_tool_call_ids:
            continue
        raise ValueError(f"Unsupported event type in ACP trajectory: {type(event).__name__}")

    return "\n\n".join(rendered_events)


def _render_tool_call(tool_name: str, arguments: object, tool_call_id: str) -> str:
    return f"{tool_name}({_truncate_rendered_event_content(str(arguments))}) [id={tool_call_id}]"


def render_trajectory_messages(
    messages: Sequence[Message],
    events: Sequence[Event],
    tool_call_id_mapping: Mapping[str, str] | None = None,
) -> str:
    action_steps_by_tool_call_id: dict[str, int] = {}
    for step_number, action, _ in index_actions(events).values():
        if action.tool_call is None:
            raise ValueError(f"Action event {action.id} is missing tool_call")
        tool_call_id = action.tool_call.id
        if tool_call_id_mapping is not None:
            mapped_tool_call_id = tool_call_id_mapping.get(tool_call_id)
            if mapped_tool_call_id is None:
                continue
            tool_call_id = mapped_tool_call_id
        if tool_call_id in action_steps_by_tool_call_id:
            raise ValueError(f"Duplicate action tool-call id: {tool_call_id}")
        action_steps_by_tool_call_id[tool_call_id] = step_number

    rendered_messages: list[str] = []
    for message in messages:
        message_action_steps = [action_steps_by_tool_call_id[tool_call.id] for tool_call in (message.tool_calls or [])]
        content_text = "\n".join(item.text for item in message.content if isinstance(item, TextContent)).strip()
        if content_text.startswith(("<summarized_action ", "<summarized_observation ")):
            rendered_messages.append(
                content_text.replace(
                    "</summarized_action>\n<summarized_", "</summarized_action>\n\n<summarized_"
                ).replace("</summarized_observation>\n<summarized_", "</summarized_observation>\n\n<summarized_")
            )
            continue

        rendered_content = content_text or "<empty>"
        if message.tool_call_id is not None:
            rendered_content = _truncate_rendered_event_content(rendered_content)
            step_number = action_steps_by_tool_call_id[message.tool_call_id]
            rendered_messages.append(
                f"<observation step={step_number} tool_call_id={message.tool_call_id}>\n"
                f"<content>\n{rendered_content}\n</content>\n"
                "</observation>"
            )
            continue

        if message.tool_calls:
            rendered_content = _truncate_rendered_event_content(rendered_content)
            if len(message_action_steps) == 1:
                action_opening_tag = f"<action step={message_action_steps[0]}>"
                rendered_tool_calls = "\n".join(
                    _render_tool_call(tool_call.name, tool_call.arguments, tool_call.id)
                    for tool_call in message.tool_calls
                )
            else:
                action_opening_tag = f"<action steps={','.join(str(step) for step in message_action_steps)}>"
                rendered_tool_calls = "\n".join(
                    f"<tool_call step={action_steps_by_tool_call_id[tool_call.id]}>\n"
                    f"{_render_tool_call(tool_call.name, tool_call.arguments, tool_call.id)}\n"
                    "</tool_call>"
                    for tool_call in message.tool_calls
                )
            rendered_messages.append(
                f"{action_opening_tag}\n"
                f"<content>\n{rendered_content}\n</content>\n"
                f"<tool_calls>\n{rendered_tool_calls}\n</tool_calls>\n"
                "</action>"
            )
            continue

        rendered_messages.append(
            f"<message role={message.role}>\n<content>\n{rendered_content}\n</content>\n</message>"
        )

    return "\n\n".join(rendered_messages)


def render_trajectory(
    events: Sequence[Event],
    *,
    state: ConversationState | None = None,
    start_at_first_action_event: bool = False,
    step_number_source_events: Sequence[Event] | None = None,
    normalize_tool_call_ids: bool = True,
) -> str:
    """Render a trajectory, optionally excluding its initial system and user messages."""
    events = get_active_branch_events(events, state)
    if any(isinstance(event, ACPToolCallEvent) for event in events):
        return render_acp_trajectory(
            events,
            start_at_first_action_event=start_at_first_action_event,
            normalize_tool_call_ids=normalize_tool_call_ids,
        )
    if start_at_first_action_event:
        first_action_event_index = next(
            (index for index, event in enumerate(events) if isinstance(event, ActionEvent)),
            None,
        )
        if first_action_event_index is None:
            raise ValueError("Cannot skip system and user messages from a trajectory without an ActionEvent")
        events = events[first_action_event_index:]
    messages = trajectory_events_to_messages(events)
    tool_call_id_mapping = None
    if normalize_tool_call_ids:
        messages, tool_call_id_mapping = remap_tool_call_ids(messages)
    return render_trajectory_messages(
        messages,
        events if step_number_source_events is None else step_number_source_events,
        tool_call_id_mapping,
    )
