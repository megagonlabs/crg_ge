import os
import tarfile
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal, overload

from openhands.sdk import LLM as OpenHandsLLM
from openhands.sdk import Tool, get_logger
from openhands.sdk.agent import Agent
from openhands.sdk.conversation import ConversationState, LocalConversation
from openhands.sdk.conversation.event_store import LOCK_FILE_NAME
from openhands.sdk.conversation.persistence_const import BASE_STATE, EVENT_NAME_RE, EVENTS_DIR
from openhands.sdk.event import ActionEvent, Event, ObservationEvent
from openhands.sdk.io import InMemoryFileStore
from openhands.tools.preset.default import get_default_tools
from pydantic import Field, PrivateAttr, SecretStr

from crg_ce.estimators.openhands.config import AgentConfig, SupportedAgentToolPreset, SupportedAgentToolSet
from crg_ce.llm_concurrency import LLMCallLimiter

ActionIndex = dict[str, tuple[int, ActionEvent, list[ObservationEvent]]]


@overload
def load_conversation_state_and_events_from_archive(
    archive_path: str | Path,
    *,
    trajectory_type: Literal["acp"],
) -> tuple[None, list[Event]]: ...


@overload
def load_conversation_state_and_events_from_archive(
    archive_path: str | Path,
    *,
    trajectory_type: None = None,
) -> tuple[ConversationState, list[Event]]: ...


@overload
def load_conversation_state_and_events_from_archive(
    archive_path: str | Path,
    *,
    trajectory_type: str,
) -> tuple[ConversationState | None, list[Event]]: ...


def load_conversation_state_and_events_from_archive(
    archive_path: str | Path,
    *,
    trajectory_type: str | None = None,
) -> tuple[ConversationState | None, list[Event]]:
    archive_path = Path(archive_path)
    if trajectory_type == "acp":
        from crg_ce.utils.openhands_trajectory import acp_messages_to_openhands_events, load_acp_jsonl_trajectory

        return None, acp_messages_to_openhands_events(load_acp_jsonl_trajectory(archive_path))
    get_default_tools()
    with tarfile.open(archive_path, "r:gz") as archive:
        # within the archive: find the conversation base_state.json
        base_state_members = [
            member
            for member in archive.getmembers()
            if member.isfile() and PurePosixPath(member.name).name == BASE_STATE
        ]
        if not base_state_members:
            raise ValueError(f"Conversation archive contains no {BASE_STATE}: {archive_path}")
        if len(base_state_members) > 1:
            raise ValueError(
                f"Conversation archive contains multiple conversations: {archive_path}; "
                f"base states={[member.name for member in base_state_members]}"
            )
        base_state_member = base_state_members[0]
        conversation_root = PurePosixPath(base_state_member.name).parent
        events_root = conversation_root / EVENTS_DIR

        base_state_file = archive.extractfile(base_state_member)
        if base_state_file is None:
            raise FileNotFoundError(f"Could not read {base_state_member.name}")

        base_state_text = base_state_file.read().decode()
        state_without_events = ConversationState.model_validate_json(base_state_text)
        file_store_contents = {BASE_STATE: base_state_text}

        # Mirror EventLog's indexed event ordering, but fail loudly on malformed
        # archives instead of warning and truncating at the first gap.
        event_members_by_index: dict[int, tarfile.TarInfo] = {}
        event_ids_by_index: dict[int, str] = {}
        seen_event_ids: dict[str, int] = {}
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.parent != events_root:
                continue

            if path.name == LOCK_FILE_NAME:
                continue

            if not member.isfile():
                raise ValueError(f"Unexpected non-file member in event log: {member.name}")

            match = EVENT_NAME_RE.match(path.name)
            if match is None:
                # note: this was previously a `continue`, openhands/sdk/conversation/event_store.py logs a warning
                raise ValueError(f"Unrecognized event file name: {member.name}")

            index: int = int(match.group("idx"))
            event_id: str = match.group("event_id")
            if index in event_members_by_index:
                raise ValueError(f"Duplicate event index {index} in archive {archive_path}")
            if event_id in seen_event_ids:
                raise ValueError(f"Duplicate event id {event_id} at indexes {seen_event_ids[event_id]} and {index}")

            event_members_by_index[index] = member
            event_ids_by_index[index] = event_id
            seen_event_ids[event_id] = index

        # make sure we aren't missing any events
        if event_members_by_index:
            expected_indexes = set(range(max(event_members_by_index) + 1))
            actual_indexes = set(event_members_by_index)
            if actual_indexes != expected_indexes:
                missing_indexes = sorted(expected_indexes - actual_indexes)
                extra_indexes = sorted(actual_indexes - expected_indexes)
                raise ValueError(
                    f"Event index gap in archive {archive_path}: missing={missing_indexes}, unexpected={extra_indexes}"
                )

        for index in sorted(event_members_by_index):
            member = event_members_by_index[index]
            event_file = archive.extractfile(member)
            if event_file is None:
                raise FileNotFoundError(f"Could not read {member.name}")
            event_text = event_file.read().decode()
            event = Event.model_validate_json(event_text)
            if event.id != event_ids_by_index[index]:
                raise ValueError(
                    f"Event payload id {event.id} does not match filename id "
                    f"{event_ids_by_index[index]} for {member.name}"
                )
            file_store_contents[f"{EVENTS_DIR}/{PurePosixPath(member.name).name}"] = event_text

    file_store = InMemoryFileStore(
        file_store_contents,
        max_size=len(file_store_contents),
        max_memory=sum(len(contents) for contents in file_store_contents.values()),
    )
    state = ConversationState.create(
        id=state_without_events.id,
        agent=state_without_events.agent,
        workspace=state_without_events.workspace,
        persistence_dir=state_without_events.persistence_dir,
        max_iterations=state_without_events.max_iterations,
        stuck_detection=state_without_events.stuck_detection,
        file_store=file_store,
    )
    events = list(state.events)

    return state, events


def index_actions(events: Sequence[Event]) -> ActionIndex:
    actions_by_id: ActionIndex = {}
    action_idx: int = 1
    for event in events:
        if isinstance(event, ActionEvent):
            if event.id in actions_by_id:
                raise ValueError(f"Duplicate action event id: {event.id}")
            actions_by_id[event.id] = (action_idx, event, [])
            action_idx += 1
        elif isinstance(event, ObservationEvent):
            action_entry = actions_by_id.get(event.action_id)
            if action_entry is None:
                raise ValueError(f"Observation event {event.id} references unknown action id: {event.action_id}")
            _, _, observation_events = action_entry
            observation_events.append(event)
    assert sorted(i for (i, _, _) in actions_by_id.values()) == list(range(1, action_idx))
    return actions_by_id


def replay_events(conversation: LocalConversation, events: Sequence[Event]) -> None:
    """Replay events into a conversation while preserving the active branch."""
    for event in events:
        conversation.state.append_event(event)


def import_needed_tools(tool_set: SupportedAgentToolSet) -> None:
    # importing is needed for de-serialization of events to work
    if tool_set == "dialogue":
        raise ValueError(
            "no longer supported: create & switch to an upgraded software-agent-sdk "
            "in which these tools are available if needed"
        )
        # get_dialogue_tools()
    elif tool_set == "default":
        get_default_tools()
    else:
        raise ValueError(f"Unsupported tool set {tool_set}")


class ConcurrencyLimitedOpenHandsLLM(OpenHandsLLM):
    """OpenHands LLM whose async calls share a run-scoped concurrency limit."""

    _llm_limiter: LLMCallLimiter | None = PrivateAttr(default=None)
    allowed_openai_params: list[str] = Field(default_factory=list)

    def _prepare_transport_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        enable_streaming: bool,
        auth_values: tuple[str | None, dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prepared_kwargs = super()._prepare_transport_kwargs(
            messages=messages,
            enable_streaming=enable_streaming,
            auth_values=auth_values,
            **kwargs,
        )
        if self.allowed_openai_params:
            prepared_kwargs["allowed_openai_params"] = self.allowed_openai_params
        return prepared_kwargs

    async def acompletion(self, *args, **kwargs):
        if self._llm_limiter is None:
            return await super().acompletion(*args, **kwargs)
        async with self._llm_limiter.slot():
            return await super().acompletion(*args, **kwargs)

    async def aresponses(self, *args, **kwargs):
        if self._llm_limiter is None:
            return await super().aresponses(*args, **kwargs)
        async with self._llm_limiter.slot():
            return await super().aresponses(*args, **kwargs)


def build_llm(
    agent_config: AgentConfig,
    *,
    llm_limiter: LLMCallLimiter | None = None,
) -> OpenHandsLLM:
    api_key: SecretStr | None = agent_config.api_key
    if api_key and api_key.get_secret_value().startswith("$"):
        if api_key.get_secret_value()[1:] in os.environ:
            api_key = SecretStr(os.environ[api_key.get_secret_value()[1:]])
        else:
            get_logger(__name__).warning("API Key starts with $ but is not in os.environ")

    llm = ConcurrencyLimitedOpenHandsLLM(
        model=agent_config.model_name,
        api_key=api_key,
        base_url=agent_config.api_base,
        top_p=agent_config.top_p,
        enable_encrypted_reasoning=agent_config.enable_encrypted_reasoning,
        # (litellm is missing max reasoning)
        reasoning_effort=agent_config.reasoning_effort,  # type: ignore
        allowed_openai_params=agent_config.allowed_openai_params,
        max_output_tokens=agent_config.max_output_tokens,
        timeout=agent_config.timeout,
        litellm_extra_body=agent_config.completion_kwargs,
    )
    llm._llm_limiter = llm_limiter
    return llm


def build_agent_tools(tools_preset: SupportedAgentToolPreset) -> list[Tool]:
    if tools_preset == "default":
        return get_default_tools()

    if tools_preset == "gsn_agentic_graph_construction":
        from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import (
            DivideAndConquerTool,
            ParticularizeTool,
        )

        return [Tool(name=DivideAndConquerTool.name), Tool(name=ParticularizeTool.name)]

    if tools_preset == "gsn_agentic_graph_construction_interp":
        from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import (
            DivideAndConquerInterpTool,
            ParticularizeInterpTool,
        )

        return [Tool(name=DivideAndConquerInterpTool.name), Tool(name=ParticularizeInterpTool.name)]

    if tools_preset == "gsn_agentic_gather_evidence_v1":
        from crg_ce.estimators.openhands.tools.gsn.agentic_graph_tools import GatherEvidenceForAgenticGraphToolV1

        return [Tool(name=GatherEvidenceForAgenticGraphToolV1.name)]

    raise ValueError(f"Unsupported agent tools preset {tools_preset}")


def build_agent(
    agent_config: AgentConfig,
    *,
    system_prompt: str | None = None,
    llm_limiter: LLMCallLimiter | None = None,
) -> Agent:
    return Agent(
        llm=build_llm(agent_config, llm_limiter=llm_limiter),
        tools=build_agent_tools(agent_config.tools_preset),
        system_prompt=system_prompt,
        # system_prompt_filename=system_prompt_absolute_path,
        # system_prompt_kwargs={"cli_mode": True, "instance": instance_info_dict}
    )
