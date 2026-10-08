from pathlib import Path

import pytest
from openhands.sdk.event import ActionEvent
from openhands.sdk.event.llm_convertible.observation import ObservationEvent

from crg_ce.graph.graph_generators.gsn.event_summarizers import summarize_action, summarize_observation
from crg_ce.utils.openhands import index_actions, load_conversation_state_and_events_from_archive

ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/agronholm__typeguard.b6a7e438.combine_module__tr31kstl.tar.gz")
TrajectorySteps = dict[int, tuple[ActionEvent, list[ObservationEvent]]]


@pytest.fixture(scope="module")
def trajectory_steps() -> TrajectorySteps:
    if not ARCHIVE_PATH.is_file():
        pytest.skip("These tests require a locally supplied trajectory fixture.")
    _, events = load_conversation_state_and_events_from_archive(ARCHIVE_PATH)
    return {step: (action, observations) for step, action, observations in index_actions(events).values()}


def test_terminal_summary_removes_command_echo_and_retains_result_ends(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This verifies normalized heredoc echoes are removed while both ends of the actual result survive.
    action, observations = trajectory_steps[30]

    summary = summarize_observation(action, observations[0])

    assert summary is not None
    assert "python3 << 'EOF'" not in summary
    assert "Optional[str]:" in summary
    assert "get_origin: typing.Union" in summary
    assert "repr(None):\n  Result: None" in summary
    assert "Exit code: 0" in summary


def test_file_editor_action_summary_omits_file_content(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This verifies file-editor actions need no OpenHands summary and retain only their operation and path.
    action, _ = trajectory_steps[27]
    action = action.model_copy(update={"summary": None})

    assert summarize_action(action) == "file_editor create: /workspace/test_edge_cases.py"


def test_think_action_summary_remains_unchanged(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This guarantees action-specific summarization does not alter existing OpenHands thought summaries.
    action, _ = trajectory_steps[1]

    assert summarize_action(action) == action.summary


def test_missing_action_summary_uses_tool_call_arguments(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This guarantees historical actions without OpenHands metadata retain the payload used in an unsummarized replay.
    action, _ = trajectory_steps[30]
    action = action.model_copy(update={"summary": None})

    assert action.tool_call is not None
    assert summarize_action(action) == f"Tool: {action.tool_name}\nArguments: {action.tool_call.arguments}"


def test_terminal_summary_retains_head_and_tail_for_long_output(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This guarantees long terminal results use 150-character head/tail excerpts rather than a prefix alone.
    action, observations = trajectory_steps[28]

    summary = summarize_observation(action, observations[0])

    assert summary is not None
    assert "Testing edge cases" in summary
    assert "\n...\n" in summary
    assert "Edge case testing complete" in summary


def test_empty_search_summary_reports_no_matches(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This verifies a successful empty search is distinguished from an arbitrary command with no output.
    action, observations = trajectory_steps[9]

    assert summarize_observation(action, observations[0]) == "Tool: terminal\nNo matches; exit code 0."


def test_file_editor_summary_reports_change_metadata_without_preview(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This verifies file edits retain operation, path, status, and computed lines while omitting source previews.
    action, observations = trajectory_steps[23]

    assert summarize_observation(action, observations[0]) == (
        "Tool: file_editor\n"
        "Operation: str_replace\n"
        "Path: /workspace/agronholm__typeguard.b6a7e438/src/typeguard/_utils.py\n"
        "Status: success\n"
        "Changed lines: 84-89"
    )


def test_think_summary_retains_short_acknowledgement(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This guarantees short think acknowledgements remain represented and preserve one output per input event.
    action, observations = trajectory_steps[1]

    assert summarize_observation(action, observations[0]) == ("Tool: think\nResult: Your thought has been logged.")


def test_unknown_tool_uses_generic_summary(
    trajectory_steps: TrajectorySteps,
) -> None:
    # This verifies unregistered tool types remain represented through the global generic fallback.
    action, observations = trajectory_steps[1]
    unknown_observation = observations[0].model_copy(update={"tool_name": "custom_tool"})

    assert summarize_observation(action, unknown_observation) == (
        "Tool: custom_tool\nResult: Your thought has been logged."
    )
