from pathlib import Path

import pytest

from crg_ce.estimators.base_estimator import ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.openhands.replay_estimator import ReplayConfidenceEstimator, ReplayEstimatorConfig


def _ce_input(tmp_path: Path) -> ConfEstimationInput:
    archive_path = tmp_path / "conversation.tar.gz"
    archive_path.write_bytes(b"archive")
    return ConfEstimationInput(
        conversation_archive_path=archive_path,
        output_dir=tmp_path / "target-output",
        instance_id="test-instance",
        model="test-model",
        problem_statement="test problem",
    )


def test_replay_estimator_loads_the_matching_source_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This verifies replay keys source outputs by both instance and trajectory model and preserves their full payload.
    monkeypatch.chdir(tmp_path)
    source_run = Path("runs/source-run.yaml")
    source_output = Path("outputs/runs/source-run/test-instance/test-model/output.json")
    source_run.parent.mkdir()
    source_run.write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    source_output.parent.mkdir(parents=True)
    expected = ConfEstimationOutput(confidence=0.73, total_tokens=42)
    source_output.write_text(expected.model_dump_json())

    estimator = ReplayConfidenceEstimator(ReplayEstimatorConfig(replay_from=source_run))
    output = estimator.estimate_confidence(_ce_input(tmp_path))

    assert output == expected


def test_replay_estimator_rejects_missing_source_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # This verifies absent replay records raise so batch execution records the item in failures.json.
    monkeypatch.chdir(tmp_path)
    estimator = ReplayConfidenceEstimator(ReplayEstimatorConfig(replay_from=Path("runs/source-run.yaml")))

    with pytest.raises(FileNotFoundError, match="Replay output does not exist"):
        estimator.estimate_confidence(_ce_input(tmp_path))
