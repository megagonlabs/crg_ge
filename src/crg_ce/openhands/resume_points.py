from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from openhands.sdk import Event
from openhands.sdk.event import ActionEvent, AgentErrorEvent, ObservationEvent


@dataclass(frozen=True)
class AfterKActions:
    """Select the prefix ending after k complete action/observation pairs."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 0 and self.k != -1:
            raise ValueError("k must be non-negative (or -1)")

    def __call__(self, events: Sequence[Event]) -> list[Event]:
        if self.k == 0:
            return self._events_before_first_action(events)
        if self.k == -1:
            return list(events)

        selected_action_ids: list[str] = []
        seen_action_ids: set[str] = set()
        action_ids_by_tool_call_id: dict[str, str] = {}
        observed_selected_action_ids: set[str] = set()
        cutoff: int | None = None

        for index, event in enumerate(events):
            if isinstance(event, ActionEvent):
                seen_action_ids.add(event.id)
                action_ids_by_tool_call_id[event.tool_call_id] = event.id
                if len(selected_action_ids) < self.k:
                    selected_action_ids.append(event.id)
                continue

            if isinstance(event, ObservationEvent):
                if event.action_id not in seen_action_ids:
                    raise ValueError(f"Observation event {event.id} references unknown action id: {event.action_id}")
                if event.action_id in selected_action_ids:
                    observed_selected_action_ids.add(event.action_id)
                    cutoff = index + 1
                continue

            if isinstance(event, AgentErrorEvent):
                action_id = action_ids_by_tool_call_id.get(event.tool_call_id)
                if action_id is None:
                    raise ValueError(
                        f"Agent error event {event.id} references unknown tool call id: {event.tool_call_id}"
                    )
                if action_id in selected_action_ids:
                    observed_selected_action_ids.add(action_id)
                    cutoff = index + 1

        if len(selected_action_ids) < self.k:
            return list(events)

        missing_observations = [
            action_id for action_id in selected_action_ids if action_id not in observed_selected_action_ids
        ]
        if missing_observations:
            raise ValueError(f"Missing observations for selected actions: {missing_observations}")

        if cutoff is None:
            raise ValueError("Selected actions were observed, but no cutoff was found")
        return list(events[:cutoff])

    @staticmethod
    def _events_before_first_action(events: Sequence[Event]) -> list[Event]:
        for index, event in enumerate(events):
            if isinstance(event, ActionEvent):
                return list(events[:index])
            if isinstance(event, ObservationEvent):
                raise ValueError(f"Observation event {event.id} references no prior action in selected prefix")
        return list(events)


class ResumePoint(Protocol):
    def __call__(self, events: Sequence[Event]) -> Sequence[Event]: ...


class ResumePoints:
    @staticmethod
    def at_end(events: Sequence[Event]) -> list[Event]:
        return list(events)

    @staticmethod
    def after_k_actions(k: int) -> AfterKActions:
        """Return a pickleable resume-point selector for child-process runs."""
        return AfterKActions(k)
