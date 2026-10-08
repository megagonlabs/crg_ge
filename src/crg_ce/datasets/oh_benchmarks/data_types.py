from typing import NotRequired, TypedDict


# full element of huggingface dataset such as "Brendan/openhands_ce_data"
class OpenHandsCEDataPoint(TypedDict):
    instance_id: str
    patch: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    image_name: str
    repo: str
    model: str
    problem_statement: str
    instruction: str
    num_actions: int
    conversation_archive_path: str
    benchmark_run_name: str
    model_pred_patch: str
    eval_report: str
    eval_test_output: str
    resolved: bool
    benchmark: NotRequired[str]
    trajectory_type: NotRequired[str]


class ConfEstimatedOpenHandsCEDataPoint(OpenHandsCEDataPoint):
    estimated_confidence: float
    total_tokens: int
    generated_tokens: int
    cost: float
