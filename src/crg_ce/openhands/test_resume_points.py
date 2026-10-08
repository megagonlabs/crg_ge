import pytest
from openhands.sdk import Observation, TextContent
from openhands.sdk.event import ActionEvent, AgentErrorEvent, ObservationEvent
from openhands.sdk.llm import MessageToolCall

from crg_ce.openhands.resume_points import ResumePoints


class _TestObservation(Observation):
    @property
    def to_llm_content(self) -> list[TextContent]:
        return [TextContent(text="ok")]


def _action_event(label: str, *, llm_response_id: str | None = None) -> ActionEvent:
    return ActionEvent(
        thought=[],
        action=None,
        tool_name=f"tool_{label}",
        tool_call_id=f"call_{label}",
        tool_call=MessageToolCall(id=f"call_{label}", name=f"tool_{label}", arguments="{}", origin="completion"),
        llm_response_id=llm_response_id or f"response_{label}",
    )


def _observation_event(action: ActionEvent) -> ObservationEvent:
    return ObservationEvent(
        observation=_TestObservation(),
        action_id=action.id,
        tool_name=action.tool_name,
        tool_call_id=action.tool_call_id,
    )


def _agent_error_event(action: ActionEvent) -> AgentErrorEvent:
    return AgentErrorEvent(
        error="Error validating tool",
        tool_name=action.tool_name,
        tool_call_id=action.tool_call_id,
    )


def test_after_k_actions_returns_prefix_through_selected_action_observations() -> None:
    first = _action_event("first")
    second = _action_event("second")
    events = [first, _observation_event(first), second, _observation_event(second)]

    prefix = ResumePoints.after_k_actions(1)(events)

    assert prefix == events[:2]


def test_after_k_actions_zero_returns_events_before_first_action() -> None:
    first = _action_event("first")
    second = _action_event("second")
    events = [first, _observation_event(first), second]

    assert ResumePoints.after_k_actions(0)(events) == []


def test_after_k_actions_returns_all_events_when_fewer_than_k_actions_exist() -> None:
    action = _action_event("first")
    events = [action, _observation_event(action)]

    assert ResumePoints.after_k_actions(10)(events) == events


def test_after_k_actions_rejects_orphan_observation() -> None:
    action = _action_event("first")
    orphan = _observation_event(action)

    with pytest.raises(ValueError, match="unknown action id"):
        ResumePoints.after_k_actions(1)([orphan])


def test_after_k_actions_treats_agent_error_as_selected_action_response() -> None:
    # This verifies tool validation errors count as visible tool responses when selecting a replay prefix.
    first = _action_event("first")
    second = _action_event("second")
    events = [first, _agent_error_event(first), second, _observation_event(second)]

    assert ResumePoints.after_k_actions(1)(events) == events[:2]


def test_after_k_actions_rejects_orphan_agent_error() -> None:
    # This verifies agent errors must still point to an earlier action by tool_call_id.
    action = _action_event("first")
    orphan = _agent_error_event(action)

    with pytest.raises(ValueError, match="unknown tool call id"):
        ResumePoints.after_k_actions(1)([orphan])


def test_after_k_actions_rounds_up_when_next_action_precedes_selected_observation() -> None:
    first = _action_event("first", llm_response_id="response")
    second = _action_event("second", llm_response_id="response")
    events = [first, second, _observation_event(first), _observation_event(second)]

    assert ResumePoints.after_k_actions(1)(events) == events[:3]


def test_after_k_actions_rejects_missing_observation() -> None:
    action = _action_event("first")

    with pytest.raises(ValueError, match="Missing observations"):
        ResumePoints.after_k_actions(1)([action])
