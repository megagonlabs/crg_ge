from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from scipy.special import logit

from crg_ce.estimators.base_estimator import BaseConfidenceEstimator, ConfEstimationInput, ConfEstimationOutput
from crg_ce.estimators.calibration import (
    BRIER_TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME,
    TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME,
    UNCALIBRATED_OUTPUT_FILENAME,
    CalibratedConfidenceEstimator,
    CalibrationConfig,
    TemperatureCalibrationParameters,
    TemperatureCalibrator,
    build_calibrator,
    temperature_scale_confidence,
)
from crg_ce.estimators.openhands.config import output_dir_for_run_config


class _ConstantEstimator(BaseConfidenceEstimator):
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        output = ConfEstimationOutput(confidence=self.confidence, total_tokens=10)
        self.save_output(output, ce_input.output_dir)
        return output


def _calibration_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.chdir(tmp_path)
    run_config_path = Path("runs/calibration-run.yaml")
    run_config_path.parent.mkdir()
    run_config_path.write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    output_dir = output_dir_for_run_config(run_config_path)
    output_dir.mkdir(parents=True)
    return run_config_path, output_dir


def _ce_input(tmp_path: Path) -> ConfEstimationInput:
    archive_path = tmp_path / "conversation.tar.gz"
    archive_path.write_bytes(b"archive")
    return ConfEstimationInput(
        conversation_archive_path=archive_path,
        output_dir=tmp_path / "item-output",
        instance_id="test-instance",
        model="test-model",
        problem_statement="test problem",
    )


def test_calibration_config_defaults_to_no_calibration() -> None:
    # This guarantees calibration is opt-in and existing estimator configs retain their prior behavior.
    config = CalibrationConfig()

    assert config.mode == "none"
    assert config.calibrate_from is None
    assert config.allow_learn is False
    assert config.temperature is None
    assert config.learning_objective is None


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "none", "calibrate_from": "run.yaml"},
        {"mode": "none", "allow_learn": True},
        {"mode": "none", "temperature": 2.0},
        {"mode": "none", "learning_objective": "brier"},
        {"mode": "temperature"},
        {"mode": "temperature", "temperature": 2.0, "calibrate_from": "run.yaml"},
        {"mode": "temperature", "temperature": 2.0, "allow_learn": True},
        {"mode": "temperature", "temperature": 2.0, "learning_objective": "brier"},
    ],
)
def test_calibration_config_rejects_incoherent_fields(config: dict[str, object]) -> None:
    # This guarantees a configured calibration mode has an unambiguous parameter source and learning policy.
    with pytest.raises(ValidationError):
        CalibrationConfig.model_validate(config)


def test_build_temperature_calibrator_scales_final_confidence() -> None:
    # This verifies generic temperature calibration divides the final-confidence logit by the configured temperature.
    calibrator = build_calibrator(CalibrationConfig(mode="temperature", temperature=2.0))

    assert isinstance(calibrator, TemperatureCalibrator)
    assert calibrator.calibrate(0.8) == pytest.approx(2 / 3)
    assert temperature_scale_confidence(0.0, 2.0) == 0.0
    assert temperature_scale_confidence(1.0, 2.0) == 1.0


def test_build_temperature_calibrator_learns_and_reuses_nll_optimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies inverse-temperature NLL fitting persists the learned scalar and can reload it without source data.
    run_config_path, output_dir = _calibration_run(tmp_path, monkeypatch)
    pd.DataFrame(
        {
            "estimated_confidence": [0.8] * 4 + [0.2] * 4 + [1.0],
            "resolved": [True, True, True, False, True, False, False, False, True],
        }
    ).to_csv(output_dir / "results.csv", index=False)
    config = CalibrationConfig(mode="temperature", calibrate_from=run_config_path, allow_learn=True)

    learned = build_calibrator(config)

    assert isinstance(learned, TemperatureCalibrator)
    expected_temperature = float(logit(0.8) / logit(0.75))
    assert learned.temperature == pytest.approx(expected_temperature, rel=1e-5)
    parameters_path = output_dir / TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME
    persisted = TemperatureCalibrationParameters.model_validate_json(parameters_path.read_text())
    assert persisted.temperature == learned.temperature

    (output_dir / "results.csv").unlink()
    loaded = build_calibrator(config.model_copy(update={"allow_learn": False}))
    assert isinstance(loaded, TemperatureCalibrator)
    assert loaded.temperature == learned.temperature


def test_learned_temperature_rejects_incorrect_deterministic_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This guarantees fitting fails when exact endpoint probabilities make finite-temperature NLL irreparable.
    run_config_path, output_dir = _calibration_run(tmp_path, monkeypatch)
    pd.DataFrame({"estimated_confidence": [0.0, 0.5, 0.75], "resolved": [True, False, True]}).to_csv(
        output_dir / "results.csv", index=False
    )

    with pytest.raises(ValueError, match="incorrect deterministic confidences"):
        build_calibrator(CalibrationConfig(mode="temperature", calibrate_from=run_config_path, allow_learn=True))


def test_build_temperature_calibrator_learns_and_persists_brier_optimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This verifies Brier fitting finds the finite optimum and stores it separately from the NLL fit.
    run_config_path, output_dir = _calibration_run(tmp_path, monkeypatch)
    pd.DataFrame(
        {
            "estimated_confidence": [0.8] * 4 + [0.2] * 4,
            "resolved": [True, True, True, False, True, False, False, False],
        }
    ).to_csv(output_dir / "results.csv", index=False)
    config = CalibrationConfig(
        mode="temperature",
        calibrate_from=run_config_path,
        allow_learn=True,
        learning_objective="brier",
    )

    learned = build_calibrator(config)

    assert isinstance(learned, TemperatureCalibrator)
    expected_temperature = float(logit(0.8) / logit(0.75))
    assert learned.temperature == pytest.approx(expected_temperature, rel=1e-4)
    parameters_path = output_dir / BRIER_TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME
    persisted = TemperatureCalibrationParameters.model_validate_json(parameters_path.read_text())
    assert persisted.temperature == learned.temperature
    assert not (output_dir / TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME).exists()


def test_learned_temperature_rejects_an_infinite_temperature_optimum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This guarantees an NLL optimum on the excluded T=infinity boundary fails instead of persisting a huge float.
    run_config_path, output_dir = _calibration_run(tmp_path, monkeypatch)
    pd.DataFrame({"estimated_confidence": [0.9, 0.8, 0.2, 0.1], "resolved": [False, False, True, True]}).to_csv(
        output_dir / "results.csv", index=False
    )

    with pytest.raises(ValueError, match="T approaches infinity"):
        build_calibrator(CalibrationConfig(mode="temperature", calibrate_from=run_config_path, allow_learn=True))


def test_calibrated_estimator_preserves_raw_output_and_recalibrates_on_resume(tmp_path: Path) -> None:
    # This verifies calibration preserves raw confidence and resume always derives output.json from that raw value.
    estimator = CalibratedConfidenceEstimator(_ConstantEstimator(0.8), TemperatureCalibrator(2.0))
    ce_input = _ce_input(tmp_path)
    expected_confidence = 2 / 3

    output = estimator.estimate_confidence(ce_input)

    assert output.confidence == pytest.approx(expected_confidence)
    raw_output_path = ce_input.output_dir / UNCALIBRATED_OUTPUT_FILENAME
    assert ConfEstimationOutput.model_validate_json(raw_output_path.read_text()).confidence == 0.8
    (ce_input.output_dir / "output.json").write_text(ConfEstimationOutput(confidence=0.01).model_dump_json())

    resumed_output = estimator.get_saved_output(ce_input.output_dir)

    assert resumed_output is not None
    assert resumed_output.confidence == pytest.approx(expected_confidence)
    assert resumed_output.total_tokens == 10
    assert ConfEstimationOutput.model_validate_json((ce_input.output_dir / "output.json").read_text()) == resumed_output
