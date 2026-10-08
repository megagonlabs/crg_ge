import re
from collections.abc import Callable
from difflib import SequenceMatcher
from typing import Protocol, cast

from openhands.sdk.event import ActionEvent
from openhands.sdk.event.llm_convertible.observation import ObservationBaseEvent, ObservationEvent
from openhands.sdk.llm import content_to_str

TERMINAL_EXCERPT_CHARS = 150
GENERIC_SUMMARY_CHARS = 300

ObservationSummarizer = Callable[[ActionEvent, ObservationEvent], str]
ActionSummarizer = Callable[[ActionEvent], str]


class _FileEditorAction(Protocol):
    command: str
    path: str


def _summarize_file_editor_action(event: ActionEvent) -> str:
    action = cast(_FileEditorAction, event.action)
    arguments = getattr(action, "arguments", {})
    operation = getattr(action, "command", arguments.get("operation", "unknown"))
    path = getattr(action, "path", arguments.get("path", "unknown"))
    return f"file_editor {operation}: {path}"


ACTION_SUMMARIZERS: dict[str, ActionSummarizer] = {
    "file_editor": _summarize_file_editor_action,
}


def summarize_action(event: ActionEvent) -> str:
    """Summarize an uncited action, retaining its tool-call payload when no summary exists."""
    summarizer = ACTION_SUMMARIZERS.get(event.tool_name)
    if summarizer is not None:
        return summarizer(event)
    if event.summary is not None:
        return event.summary
    if event.tool_call is None:
        raise ValueError(f"Action event {event.id} is missing both a summary and tool_call")
    return f"Tool: {event.tool_name}\nArguments: {event.tool_call.arguments}"


def _observation_text(event: ObservationBaseEvent) -> str:
    return "".join(content_to_str(event.to_llm_message().content)).strip()


def _observation_result_text(event: ObservationEvent) -> str:
    return "".join(content_to_str(event.observation.content)).strip()


def _truncate_prefix(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[:limit].rstrip()}..."


def _head_and_tail(text: str) -> str:
    if len(text) <= TERMINAL_EXCERPT_CHARS * 2:
        return text
    return f"{text[:TERMINAL_EXCERPT_CHARS].rstrip()}\n...\n{text[-TERMINAL_EXCERPT_CHARS:].lstrip()}"


def summarize_acp_action(
    tool_name: str,
    title: str,
    rendered_input: str,
    *,
    action_summary: str | None = None,
    rendered_command: str | None = None,
    file_editor_operation: str | None = None,
    file_editor_path: str | None = None,
) -> str:
    """Summarize one ACP tool invocation without retaining an unbounded input payload."""
    if file_editor_operation is not None:
        if file_editor_path is None:
            raise ValueError("An ACP file-editor operation requires a path")
        return f"{tool_name} {file_editor_operation}: {file_editor_path}"
    if action_summary:
        return _truncate_prefix(action_summary, GENERIC_SUMMARY_CHARS)
    if rendered_command is not None:
        return f"Tool: {tool_name}\nCommand: {_truncate_prefix(rendered_command, GENERIC_SUMMARY_CHARS)}"
    if title:
        return f"Tool: {tool_name}\nTitle: {_truncate_prefix(title, GENERIC_SUMMARY_CHARS)}"
    return f"Tool: {tool_name}\nInput: {_truncate_prefix(rendered_input, GENERIC_SUMMARY_CHARS)}"


def summarize_acp_observation(
    tool_name: str,
    status: str | None,
    rendered_output: str,
    *,
    file_edits: list[tuple[str, str | None, str]] | None = None,
    file_editor_operation: str | None = None,
    file_editor_path: str | None = None,
) -> str:
    """Summarize one ACP tool result while retaining bounded excerpts from both ends."""
    if file_editor_operation is not None:
        if file_editor_path is None:
            raise ValueError("An ACP file-editor operation requires a path")
        if file_edits is None:
            return _file_editor_summary(
                tool_name=tool_name,
                operation=file_editor_operation,
                path=file_editor_path,
                is_error=status == "failed",
                error_result=rendered_output,
            )
        summaries: list[str] = []
        for path, old_text, new_text in file_edits:
            summaries.append(
                _file_editor_summary(
                    tool_name=tool_name,
                    operation=file_editor_operation,
                    path=path,
                    is_error=status == "failed",
                    old_content=old_text,
                    new_content=new_text,
                    error_result=rendered_output,
                )
            )
        return "\n\n".join(summaries)
    return f"Tool: {tool_name}\nStatus: {status or 'unknown'}\nOutput:\n{_head_and_tail(rendered_output)}"


def _remove_command_echo(output: str, command: str | None) -> str:
    if not command:
        return output
    command_lines = [line for line in command.splitlines() if line.strip()]
    output_lines = output.splitlines()
    if output_lines[: len(command_lines)] == command_lines:
        return "\n".join(output_lines[len(command_lines) :]).lstrip()
    return output


def _summarize_terminal(action: ActionEvent, event: ObservationEvent) -> str:
    observation = event.observation
    output = _observation_result_text(event)
    command = getattr(observation, "command", None)
    output = _remove_command_echo(output, command)

    exit_code = getattr(observation, "exit_code", None)
    exit_status = f"exit code {exit_code}" if exit_code is not None else "exit code unavailable"
    action_command = getattr(action.action, "command", "")
    if not output:
        if re.search(r"\b(?:find|grep|rg|ag|ack)\b", action_command):
            return f"Tool: terminal\nNo matches; {exit_status}."
        return f"Tool: terminal\nNo output; {exit_status}."

    return f"Tool: terminal\nExit code: {exit_code}\nOutput:\n{_head_and_tail(output)}"


def _changed_line_range(old_content: str | None, new_content: str | None) -> str | None:
    if new_content is None:
        return None

    old_lines = [] if old_content is None else old_content.splitlines()
    new_lines = new_content.splitlines()
    changed_lines: list[int] = []
    for tag, _old_start, _old_end, new_start, new_end in SequenceMatcher(None, old_lines, new_lines).get_opcodes():
        if tag == "equal":
            continue
        if new_start == new_end:
            changed_lines.append(max(1, min(new_start + 1, len(new_lines))))
        else:
            changed_lines.extend(range(new_start + 1, new_end + 1))

    if not changed_lines:
        return None
    first, last = min(changed_lines), max(changed_lines)
    return str(first) if first == last else f"{first}-{last}"


def _file_editor_summary(
    *,
    tool_name: str = "file_editor",
    operation: str,
    path: str,
    is_error: bool,
    old_content: str | None = None,
    new_content: str | None = None,
    error_result: str = "",
) -> str:
    lines = [
        f"Tool: {tool_name}",
        f"Operation: {operation}",
        f"Path: {path}",
        f"Status: {'failure' if is_error else 'success'}",
    ]
    changed_line_range = _changed_line_range(old_content, new_content)
    if changed_line_range is not None:
        lines.append(f"Changed lines: {changed_line_range}")
    if is_error:
        lines.append(f"Result: {_truncate_prefix(error_result, GENERIC_SUMMARY_CHARS)}")
    return "\n".join(lines)


def _summarize_file_editor(action: ActionEvent, event: ObservationEvent) -> str:
    observation = event.observation
    arguments = getattr(action.action, "arguments", {})
    operation = getattr(
        action.action,
        "command",
        arguments.get("operation", getattr(observation, "command", "unknown")),
    )
    path = getattr(action.action, "path", arguments.get("path", getattr(observation, "path", "unknown")))
    is_error = bool(getattr(observation, "is_error", False))
    return _file_editor_summary(
        operation=operation,
        path=path,
        is_error=is_error,
        old_content=getattr(observation, "old_content", None),
        new_content=getattr(observation, "new_content", None),
        error_result=_observation_result_text(event),
    )


OBSERVATION_SUMMARIZERS: dict[str, ObservationSummarizer] = {
    "terminal": _summarize_terminal,
    "file_editor": _summarize_file_editor,
}


def summarize_observation(action: ActionEvent, event: ObservationBaseEvent) -> str:
    """Summarize an uncited observation, using a generic fallback for unknown tools."""
    if isinstance(event, ObservationEvent):
        summarizer = OBSERVATION_SUMMARIZERS.get(event.tool_name)
        if summarizer is not None:
            return summarizer(action, event)

    result = _truncate_prefix(_observation_text(event), GENERIC_SUMMARY_CHARS)
    return f"Tool: {event.tool_name}\nResult: {result}"
