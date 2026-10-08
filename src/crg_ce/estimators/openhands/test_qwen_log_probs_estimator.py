from pathlib import Path
from typing import cast

import pytest

from crg_ce.estimators.base_estimator import ConfEstimationInput
from crg_ce.estimators.openhands.config import LogProbsEstimatorConfig
from crg_ce.estimators.openhands.log_probs_estimator import (
    LogProbFeatures,
    LogProbsEstimator,
    _action_step_token_indexes,
)

SURROGATE_MODEL = "openai/Qwen/Qwen3.8-27B"
FIXTURE_DIR = Path("src/crg_ce/estimators/openhands/test_data/qwen_log_probs_estimator")
ARCHIVE_PATH = FIXTURE_DIR / "offer_letter_generator.tar.gz"
REQUEST_PATH = FIXTURE_DIR / "log_prob_request.json"
FEATURES_PATH = FIXTURE_DIR / "log_prob_features.json"
LAST_ACTION_SEQUENCE_PATH = FIXTURE_DIR / "last_action_sequence.txt"
FIRST_ACTION_SEQUENCE_PATH = FIXTURE_DIR / "first_action_sequence.txt"
HIDDEN_REASONING_PREFIX = "Let's start by exploring the files."
VISIBLE_ACTION_THOUGHT = "The table cells show minimal text — placeholders may be split across runs."

def test_qwen_raw_and_length_normalized_minima_select_different_real_actions(tmp_path: Path) -> None:
    # This uses the same real Qwen fixture to guarantee raw sequence probability selects the long file-creation
    # action, while length normalization selects a shorter lower-per-token-confidence verification action.
    if not ARCHIVE_PATH.is_file() or not FEATURES_PATH.is_file():
        pytest.skip("This test requires locally supplied trajectory and log-probability fixtures.")
    config = LogProbsEstimatorConfig.model_validate(
        {"agent": {"model_name": "openai/Qwen/Qwen3.8-27B", "api_key": "local-key"}}
    )
    estimator_input = ConfEstimationInput(
        conversation_archive_path=ARCHIVE_PATH,
        output_dir=tmp_path,
        instance_id="skills-openai-Qwen-Qwen3.8-27B-FP8-oh-offer-letter-generator",
        model="openai/Qwen/Qwen3.8-27B-FP8",
        problem_statement="Fill the provided offer-letter template.",
        benchmark="skillsbench",
    )
    messages, _ = LogProbsEstimator(config)._prompt(estimator_input)
    features = LogProbFeatures.model_validate_json(FEATURES_PATH.read_text())
    action_steps = _action_step_token_indexes(messages, features, SURROGATE_MODEL)
    action_log_probs: list[list[float]] = [
        [cast(float, features.token_log_probs[index]) for index in step] for step in action_steps
    ]
    raw_min_index = min(range(len(action_steps)), key=lambda index: sum(action_log_probs[index]))
    normalized_min_index = min(
        range(len(action_steps)),
        key=lambda index: sum(action_log_probs[index]) / len(action_log_probs[index]),
    )
    raw_min_sequence = "".join(features.tokens[index] or "" for index in action_steps[raw_min_index])
    normalized_min_sequence = "".join(features.tokens[index] or "" for index in action_steps[normalized_min_index])

    assert raw_min_index == 4
    assert "<parameter=path>\n/root/fill_offer_letter.py\n</parameter>" in raw_min_sequence
    assert normalized_min_index == 5
    assert "check the python-docx version" in normalized_min_sequence
