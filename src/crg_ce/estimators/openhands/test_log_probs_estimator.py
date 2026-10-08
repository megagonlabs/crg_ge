import json
import math
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, TypedDict, cast

import pytest
from litellm.types.utils import Choices, ModelResponse
from litellm.types.utils import Message as LiteLLMMessage
from openhands.sdk import LLM, Agent, Event, LocalConversation, LocalWorkspace
from openhands.sdk.event import SystemPromptEvent

from crg_ce.estimators.base_estimator import ConfEstimationInput, ConfEstimationOutput, ModelUsage
from crg_ce.estimators.openhands.config import LogProbsEstimatorConfig
from crg_ce.estimators.openhands.log_probs_estimator import (
    LogProbFeatures,
    LogProbsEstimator,
    _action_step_token_indexes,
    _aggregate_action_probability,
    _last_action_mean_log_prob,
)
from crg_ce.utils.openhands import load_conversation_state_and_events_from_archive, replay_events

SURROGATE_MODEL = "openai/Qwen/Qwen3.8-27B"
ARCHIVE_PATH = Path("src/crg_ce/utils/test_data/agronholm__typeguard.b6a7e438.combine_module__tr31kstl.tar.gz")
LAST_ACTION_ARCHIVE_PATH = Path(
    "src/crg_ce/estimators/openhands/test_data/qwen_log_probs_estimator/offer_letter_generator.tar.gz"
)
LOG_PROB_FEATURES_PATH = Path(
    "src/crg_ce/estimators/openhands/test_data/qwen_log_probs_estimator/log_prob_features.json"
)


class _SteppableAgent(Protocol):
    def step(self, conversation: LocalConversation, on_event: Callable[[Event], None]) -> None: ...


class _OpenHandsCall(TypedDict):
    messages: list[dict[Any, Any]]
    tools: list[dict[str, Any]]


class _EstimatorCall(TypedDict):
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    extra_body: dict[str, Any]


def _minimal_model_response() -> ModelResponse:
    return ModelResponse(
        id="response-1",
        created=0,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=LiteLLMMessage(role="assistant", content="done"),
            )
        ],
    )


def test_log_probs_estimator_matches_the_prompt_submitted_by_a_real_openhands_agent(
    monkeypatch, tmp_path: Path
) -> None:
    # This verifies a real OpenHands Agent and the estimator produce text-identical chat-template inputs from the
    # complete archive. It also guarantees transient estimator failures are retried up to a successful third call.
    if not ARCHIVE_PATH.is_file():
        pytest.skip("This test requires a locally supplied trajectory fixture.")
    archived_state, events = load_conversation_state_and_events_from_archive(ARCHIVE_PATH)
    openhands_calls: list[_OpenHandsCall] = []

    def fake_openhands_completion(**kwargs: Any) -> ModelResponse:
        openhands_calls.append(cast(_OpenHandsCall, kwargs))
        return _minimal_model_response()

    monkeypatch.setattr("openhands.sdk.llm.llm.litellm_completion", fake_openhands_completion)
    agent = Agent(
        llm=LLM(model="openai/gpt-4o-mini", api_key="local-key", num_retries=0),
        tools=[],
    )
    conversation = LocalConversation(
        agent=agent,
        workspace=LocalWorkspace(working_dir=tmp_path / "workspace"),
        persistence_dir=None,
        conversation_id=archived_state.id,
        visualizer=None,
        delete_on_close=False,
    )
    agent._initialize(conversation.state)
    system_event = next(event for event in events if isinstance(event, SystemPromptEvent))
    agent._tools = {tool.name: tool for tool in system_event.tools}
    replay_events(conversation, events)
    try:
        cast(_SteppableAgent, agent).step(conversation, lambda event: None)
    finally:
        agent._tools = {}
    assert len(openhands_calls) == 1

    estimator_calls: list[_EstimatorCall] = []

    def fake_completion(**kwargs: Any) -> SimpleNamespace:
        call = cast(_EstimatorCall, kwargs)
        estimator_calls.append(call)
        if len(estimator_calls) < 3:
            raise ConnectionError("temporary surrogate failure")
        return SimpleNamespace(
            model="openai/gpt-4o-mini",
            choices=[SimpleNamespace()],
            prompt_token_ids=[101, 102, 103],
            prompt_logprobs=[
                None,
                {102: {"logprob": -0.2, "decoded_token": "hello"}},
                {103: {"logprob": -0.4, "decoded_token": " world"}},
            ],
            usage=SimpleNamespace(
                prompt_tokens=3,
                completion_tokens=1,
                total_tokens=4,
            ),
            response_cost=0.0,
        )

    monkeypatch.setattr("crg_ce.estimators.openhands.log_probs_estimator.litellm.completion", fake_completion)
    cfg = LogProbsEstimatorConfig.model_validate(
        {
            "agent": {
                "model_name": "openai/gpt-4o-mini",
                "api_base": "http://localhost:8000/v1",
                "api_key": "local-key",
            }
        }
    )
    ce_input = ConfEstimationInput(
        conversation_archive_path=ARCHIVE_PATH,
        output_dir=tmp_path / "output",
        instance_id="test-instance",
        model="trajectory-model",
        problem_statement="test problem",
    )

    output = LogProbsEstimator(cfg).estimate_confidence(ce_input)

    assert len(estimator_calls) == 3
    assert estimator_calls[-1]["messages"] == openhands_calls[0]["messages"]
    assert estimator_calls[-1]["tools"] == openhands_calls[0]["tools"]
    assert estimator_calls[-1]["extra_body"] == {"prompt_logprobs": 0, "return_token_ids": True}
    assert output.confidence == pytest.approx(math.exp(-0.3))
    assert output.model_copy(update={"confidence": 0.0}) == ConfEstimationOutput(
        confidence=0.0,
        total_tokens=4,
        generated_tokens=1,
        cost=0.0,
        usage_by_model={
            "openai/gpt-4o-mini": ModelUsage(
                calls=1,
                prompt_tokens=3,
                completion_tokens=1,
                reasoning_tokens=0,
                total_tokens=4,
                cost=0.0,
            )
        },
    )
    features = LogProbFeatures.model_validate_json((ce_input.output_dir / "log_prob_features.json").read_text())
    assert features.token_ids == [101, 102, 103]
    assert features.tokens == [None, "hello", " world"]
    assert features.token_log_probs == [None, -0.2, -0.4]
    assert features.mean_log_prob == pytest.approx(-0.3)
    assert json.loads((ce_input.output_dir / "log_prob_prompt.json").read_text()) == {
        "messages": estimator_calls[-1]["messages"],
        "tools": estimator_calls[-1]["tools"],
    }


def test_log_probs_estimator_can_request_prompt_scores_with_the_openai_client(monkeypatch) -> None:
    # This verifies the experimental OpenAI transport bypasses LiteLLM and sends the vLLM-compatible prompt-scoring
    # extensions directly. It assumes the selected OpenAI endpoint accepts those extension fields.
    client_kwargs: dict[str, Any] = {}
    request_kwargs: dict[str, Any] = {}
    expected_response = SimpleNamespace(prompt_token_ids=[1], prompt_logprobs=[None], choices=[object()])

    class FakeCompletions:
        def create(self, **kwargs: Any) -> SimpleNamespace:
            request_kwargs.update(kwargs)
            return expected_response

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            client_kwargs.update(kwargs)
            self.chat = SimpleNamespace(completions=FakeCompletions())

        def __enter__(self) -> "FakeOpenAI":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setattr("crg_ce.estimators.openhands.log_probs_estimator.OpenAI", FakeOpenAI)
    cfg = LogProbsEstimatorConfig.model_validate(
        {
            "completion_client": "openai",
            "agent": {
                "model_name": "gpt-5.6-luna",
                "api_key": "$OPENAI_API_KEY",
                "reasoning_effort": "none",
            },
        }
    )
    messages = [{"role": "user", "content": "hello"}]
    tools: list[dict[str, Any]] = []

    response = LogProbsEstimator(cfg)._complete(messages, tools)

    assert response is expected_response
    assert client_kwargs == {"api_key": "test-openai-key"}
    assert request_kwargs == {
        "model": "gpt-5.6-luna",
        "messages": messages,
        "tools": tools,
        "max_completion_tokens": 1,
        "reasoning_effort": "none",
        "timeout": 600,
        "extra_body": {"prompt_logprobs": 0, "return_token_ids": True},
    }


def test_last_action_mean_log_prob_selects_final_action_from_real_conversation(tmp_path: Path) -> None:
    if not LAST_ACTION_ARCHIVE_PATH.is_file() or not LOG_PROB_FEATURES_PATH.is_file():
        pytest.skip("This test requires locally supplied trajectory and log-probability fixtures.")
    # The saved token scores and archive come from the same offer-letter trajectory. The final assistant response
    # occupies a fixed token slice, so preceding tool results and the unfinished generation prefix must be excluded.
    cfg = LogProbsEstimatorConfig.model_validate(
        {"agent": {"model_name": SURROGATE_MODEL, "api_key": "local-key"}}
    )
    ce_input = ConfEstimationInput(
        conversation_archive_path=LAST_ACTION_ARCHIVE_PATH,
        output_dir=tmp_path,
        instance_id="test-instance",
        model="trajectory-model",
        problem_statement="test problem",
    )
    messages, _ = LogProbsEstimator(cfg)._prompt(ce_input)
    last_action_index = max(index for index, message in enumerate(messages) if message["role"] == "assistant")
    last_action = messages[last_action_index]
    final_text = last_action["content"][0]["text"]
    assert final_text.startswith("\n\nDone. The offer letter is saved at `/root/offer_letter_filled.docx`.")
    assert last_action_index == len(messages) - 1
    expected_rendered_final_action = (
        "<|im_start|>assistant\n<think>\n\n</think>" + final_text + "<|im_end|>"
    )
    features = LogProbFeatures.model_validate_json(LOG_PROB_FEATURES_PATH.read_text())
    action_token_slice = slice(16211, 16587)
    expected_action_log_probs = [
        log_prob for log_prob in features.token_log_probs[action_token_slice] if log_prob is not None
    ]

    assert "".join(token or "" for token in features.tokens[action_token_slice]) == expected_rendered_final_action
    assert len(expected_action_log_probs) == 376
    assert _last_action_mean_log_prob(messages, features, SURROGATE_MODEL) == pytest.approx(
        sum(expected_action_log_probs) / len(expected_action_log_probs)
    )


def test_action_sequence_probability_aggregations_exclude_non_assistant_messages() -> None:
    # This isolates the aggregation math from the model API. It guarantees system, user, and tool-response tokens are
    # excluded; raw and length-normalized minima can select different actions; and the legacy name is an exact alias.
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "long, confident action"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "shorter, less confident action"},
        {"role": "user", "content": "observation"},
    ]
    tokens = [
        "<|im_start|>system\nsystem<|im_end|>",
        "<|im_start|>assistant",
        "\nlong",
        " ",
        "confident",
        " ",
        "action",
        " ",
        "content",
        "\n",
        "<|im_end|>",
        "<|im_start|>user\n<tool_response>observation</tool_response><|im_end|>",
        "<|im_start|>assistant",
        "\nshort action",
        "<|im_end|>",
        "<|im_start|>user\nobservation<|im_end|>",
    ]
    token_log_probs = [
        math.log(0.01),
        *[math.log(0.9)] * 10,
        math.log(0.01),
        *[math.log(0.8)] * 3,
        math.log(0.01),
    ]
    features = LogProbFeatures(
        token_ids=list(range(len(tokens))),
        tokens=cast(list[str | None], tokens),
        token_log_probs=cast(list[float | None], token_log_probs),
        mean_log_prob=sum(token_log_probs) / len(token_log_probs),
    )

    def aggregate(aggregation_type: str) -> float:
        return _aggregate_action_probability(messages, features, aggregation_type, SURROGATE_MODEL)

    assert _action_step_token_indexes(messages, features, SURROGATE_MODEL) == [
        list(range(1, 11)),
        list(range(12, 15)),
    ]
    assert aggregate("seq_prob_first") == pytest.approx(0.9**10)
    assert aggregate("seq_prob_last") == pytest.approx(0.8**3)
    assert aggregate("seq_prob_mean") == pytest.approx((0.9**10 + 0.8**3) / 2)
    assert aggregate("seq_prob_min") == pytest.approx(0.9**10)
    assert aggregate("len_norm_seq_prob_first") == pytest.approx(0.9)
    assert aggregate("len_norm_seq_prob_mean") == pytest.approx(0.85)
    assert aggregate("len_norm_seq_prob_min") == pytest.approx(0.8)
    assert aggregate("len_norm_seq_prob_last") == pytest.approx(0.8)
    assert aggregate("last_action_mean_log_prob") == aggregate("len_norm_seq_prob_last")


def test_log_probs_estimator_replays_saved_features_without_an_llm_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This guarantees a derived aggregation loads token features and their exact source prompt by item coordinates,
    # makes no model call, reports no new usage/cost, and copies the replayable artifacts into its own output.
    monkeypatch.chdir(tmp_path)
    archive_path = tmp_path / "conversation.tar.gz"
    archive_path.write_bytes(b"unused during replay")
    source_item_dir = Path("outputs/runs/source-log-probs/test-instance/test-model")
    source_item_dir.mkdir(parents=True)
    prompt = {"messages": [{"role": "assistant", "content": "action"}], "tools": []}
    features = LogProbFeatures(
        token_ids=[1, 2, 3],
        tokens=["<|im_start|>assistant", "\naction", "<|im_end|>"],
        token_log_probs=[math.log(0.5), math.log(0.25), math.log(0.5)],
        mean_log_prob=math.log(0.5) + math.log(0.25) + math.log(0.5),
    )
    (source_item_dir / "log_prob_prompt.json").write_text(json.dumps(prompt))
    (source_item_dir / "log_prob_features.json").write_text(features.model_dump_json())
    config = LogProbsEstimatorConfig.model_validate(
        {
            "replay_from": "runs/source-log-probs.yaml",
            "aggregation_type": "seq_prob_last",
            "agent": {"model_name": "openai/Qwen/Qwen3.8-27B", "api_key": "local-key"},
        }
    )
    ce_input = ConfEstimationInput(
        conversation_archive_path=archive_path,
        output_dir=tmp_path / "derived-output",
        instance_id="test-instance",
        model="test-model",
        problem_statement="unused during replay",
    )

    def fail_completion(**kwargs: Any) -> None:
        raise AssertionError(f"Replay unexpectedly called the model with {kwargs}")

    monkeypatch.setattr("crg_ce.estimators.openhands.log_probs_estimator.litellm.completion", fail_completion)
    output = LogProbsEstimator(config).estimate_confidence(ce_input)

    assert output.confidence == pytest.approx(0.0625)
    assert output.model_copy(update={"confidence": 0.0}) == ConfEstimationOutput(
        confidence=0.0,
        total_tokens=0,
        generated_tokens=0,
        cost=0.0,
    )
    assert json.loads((ce_input.output_dir / "log_prob_prompt.json").read_text()) == prompt
    assert LogProbFeatures.model_validate_json((ce_input.output_dir / "log_prob_features.json").read_text()) == features
