import threading
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize
from scipy.special import expit, logit

from crg_ce.estimators.base_estimator import (
    BaseConfidenceEstimator,
    ConfEstimationInput,
    ConfEstimationOutput,
)

CalibrationMode = Literal["none", "temperature"]
TemperatureLearningObjective = Literal["nll", "brier"]
TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME = "temperature_calibration_parameters.json"
BRIER_TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME = "brier_temperature_calibration_parameters.json"
UNCALIBRATED_OUTPUT_FILENAME = "uncalibrated_output.json"


class CalibrationConfig(BaseModel):
    calibrate_from: Path | None = Field(
        default=None,
        description="Run-config YAML whose results.csv supplies calibration examples and stores learned parameters.",
    )
    mode: CalibrationMode = "none"
    allow_learn: bool = False
    temperature: float | None = Field(default=None, gt=0)
    learning_objective: TemperatureLearningObjective | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_mode_fields(self) -> Self:
        if self.mode == "none":
            if self.calibrate_from is not None:
                raise ValueError("calibrate_from must not be set when calibration mode is 'none'")
            if self.allow_learn:
                raise ValueError("allow_learn must be false when calibration mode is 'none'")
            if self.temperature is not None:
                raise ValueError("temperature must not be set when calibration mode is 'none'")
            if self.learning_objective is not None:
                raise ValueError("learning_objective must not be set when calibration mode is 'none'")
        elif self.mode == "temperature":
            if (self.calibrate_from is None) == (self.temperature is None):
                raise ValueError(
                    "Exactly one of calibrate_from or temperature is required when calibration mode is 'temperature'"
                )
            if self.temperature is not None and self.allow_learn:
                raise ValueError("allow_learn must be false when a fixed temperature is configured")
            if self.temperature is not None and self.learning_objective is not None:
                raise ValueError("learning_objective must not be set when a fixed temperature is configured")
        return self


def temperature_scale_confidence(confidence: float, temperature: float) -> float:
    """Apply probability temperature scaling by dividing the confidence logit by temperature."""
    if not 0 <= confidence <= 1:
        raise ValueError(f"Expected confidence in [0, 1], got {confidence}")
    if temperature <= 0:
        raise ValueError(f"Expected positive temperature, got {temperature}")
    if confidence in {0, 1}:
        return confidence
    return float(expit(logit(confidence) / temperature))


class TemperatureCalibrator:
    def __init__(self, temperature: float) -> None:
        self.temperature = temperature

    def calibrate(self, confidence: float) -> float:
        return temperature_scale_confidence(confidence, self.temperature)


class TemperatureCalibrationParameters(BaseModel):
    temperature: float = Field(gt=0)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


_PARAMS_LOCK = threading.Lock()


def _load_temperature_training_data(results_path: Path) -> tuple[np.ndarray, np.ndarray]:
    confidence_column = "estimated_confidence"
    correctness_column = "resolved"
    results = pd.read_csv(results_path)
    required_columns = {confidence_column, correctness_column}
    missing_columns = required_columns - set(results.columns)
    if missing_columns:
        raise ValueError(f"Calibration results are missing columns {sorted(missing_columns)}: {results_path}")
    if results[list(required_columns)].isnull().any().any():
        raise ValueError(f"Calibration results contain missing confidence or correctness values: {results_path}")
    if not results[confidence_column].between(0.0, 1.0).all():
        raise ValueError(f"Calibration confidences must be between 0 and 1: {results_path}")

    confidence = results[confidence_column].to_numpy(dtype=float)
    correctness = results[correctness_column].astype(int).to_numpy()
    if set(np.unique(correctness)) != {0, 1}:
        raise ValueError(f"Temperature calibration requires both resolved and unresolved examples: {results_path}")
    incorrect_deterministic = ((confidence == 0) & (correctness == 1)) | ((confidence == 1) & (correctness == 0))
    if incorrect_deterministic.any():
        raise ValueError(
            "Temperature calibration cannot repair incorrect deterministic confidences at indices "
            f"{np.flatnonzero(incorrect_deterministic).tolist()}: {results_path}"
        )

    nondeterministic = (confidence > 0) & (confidence < 1)
    logits = logit(confidence[nondeterministic])
    labels = correctness[nondeterministic]
    if logits.size == 0:
        raise ValueError(f"Temperature is unidentifiable from only deterministic confidences: {results_path}")
    return logits, labels


def _learn_temperature_nll(results_path: Path) -> float:
    logits, labels = _load_temperature_training_data(results_path)
    gradient_at_zero = np.mean((0.5 - labels) * logits)
    if gradient_at_zero >= 0:
        raise ValueError(f"Temperature calibration has no finite NLL optimum (T approaches infinity): {results_path}")
    signed_logits = (2 * labels - 1) * logits
    if (signed_logits >= 0).all():
        raise ValueError(f"Temperature calibration has no finite NLL optimum (T approaches zero): {results_path}")

    def nll_and_gradient(inverse_temperature: np.ndarray) -> tuple[float, np.ndarray]:
        scaled_logits = inverse_temperature[0] * logits
        nll = np.mean(np.logaddexp(0, scaled_logits) - labels * scaled_logits)
        gradient = np.mean((expit(scaled_logits) - labels) * logits)
        return float(nll), np.array([gradient])

    fit = minimize(
        nll_and_gradient,
        x0=np.array([1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)],
    )
    if not fit.success:
        raise RuntimeError(f"Temperature calibration optimization failed for {results_path}: {fit.message}")
    return 1.0 / float(fit.x[0])


def _learn_temperature_brier(results_path: Path) -> float:
    logits, labels = _load_temperature_training_data(results_path)

    def brier_and_gradient(inverse_temperature: np.ndarray) -> tuple[float, np.ndarray]:
        probabilities = expit(inverse_temperature[0] * logits)
        residuals = probabilities - labels
        brier = np.mean(residuals**2)
        gradient = np.mean(2 * residuals * probabilities * (1 - probabilities) * logits)
        return float(brier), np.array([gradient])

    fit = minimize(
        brier_and_gradient,
        x0=np.array([1.0]),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)],
    )
    if not fit.success:
        raise RuntimeError(f"Brier temperature calibration optimization failed for {results_path}: {fit.message}")
    inverse_temperature = float(fit.x[0])
    if inverse_temperature == 0:
        raise ValueError(f"Temperature calibration has no finite Brier optimum (T approaches infinity): {results_path}")
    return 1.0 / inverse_temperature


def build_calibrator(config: CalibrationConfig) -> TemperatureCalibrator | None:
    if config.mode == "none":
        return None
    if config.temperature is not None:
        return TemperatureCalibrator(config.temperature)
    assert config.calibrate_from is not None

    from crg_ce.estimators.openhands.config import output_dir_for_run_config

    calibration_output_dir = output_dir_for_run_config(config.calibrate_from)
    learning_objective = config.learning_objective or "nll"
    parameters_filename = (
        TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME
        if learning_objective == "nll"
        else BRIER_TEMPERATURE_CALIBRATION_PARAMETERS_FILENAME
    )
    parameters_path = calibration_output_dir / parameters_filename
    with _PARAMS_LOCK:
        if parameters_path.is_file():
            parameters = TemperatureCalibrationParameters.model_validate_json(parameters_path.read_text())
        elif config.allow_learn:
            learn_temperature = _learn_temperature_nll if learning_objective == "nll" else _learn_temperature_brier
            parameters = TemperatureCalibrationParameters(
                temperature=learn_temperature(calibration_output_dir / "results.csv")
            )
            parameters_path.write_text(parameters.model_dump_json(indent=2))
        else:
            raise FileNotFoundError(
                f"Calibration parameters do not exist and allow_learn is false: {parameters_path}"
            )
    return TemperatureCalibrator(parameters.temperature)


class CalibratedConfidenceEstimator(BaseConfidenceEstimator):
    def __init__(
        self,
        estimator: BaseConfidenceEstimator,
        calibrator: TemperatureCalibrator,
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator

    def _calibrate_output(self, output: ConfEstimationOutput) -> ConfEstimationOutput:
        return output.model_copy(update={"confidence": self.calibrator.calibrate(output.confidence)})

    def _save_uncalibrated_output(self, output: ConfEstimationOutput, item_output_dir: Path) -> None:
        item_output_dir.mkdir(exist_ok=True, parents=True)
        (item_output_dir / UNCALIBRATED_OUTPUT_FILENAME).write_text(output.model_dump_json(indent=2))

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        uncalibrated_output = self.estimator.estimate_confidence(ce_input)
        self._save_uncalibrated_output(uncalibrated_output, ce_input.output_dir)
        calibrated_output = self._calibrate_output(uncalibrated_output)
        self.save_output(calibrated_output, ce_input.output_dir)
        return calibrated_output

    async def aestimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        uncalibrated_output = await self.estimator.aestimate_confidence(ce_input)
        self._save_uncalibrated_output(uncalibrated_output, ce_input.output_dir)
        calibrated_output = self._calibrate_output(uncalibrated_output)
        self.save_output(calibrated_output, ce_input.output_dir)
        return calibrated_output

    async def aestimate_confidence_batch(
        self, ce_inputs: Sequence[ConfEstimationInput]
    ) -> Sequence[ConfEstimationOutput | BaseException]:
        uncalibrated_outputs = await self.estimator.aestimate_confidence_batch(ce_inputs)
        calibrated_outputs: list[ConfEstimationOutput | Exception] = []
        for ce_input, uncalibrated_output in zip(ce_inputs, uncalibrated_outputs, strict=True):
            if isinstance(uncalibrated_output, Exception):
                calibrated_outputs.append(uncalibrated_output)
                continue
            self._save_uncalibrated_output(uncalibrated_output, ce_input.output_dir)  # type: ignore
            calibrated_output = self._calibrate_output(uncalibrated_output)  # type: ignore
            self.save_output(calibrated_output, ce_input.output_dir)
            calibrated_outputs.append(calibrated_output)
        return calibrated_outputs

    def get_saved_output(self, item_output_dir: Path) -> ConfEstimationOutput | None:
        uncalibrated_output_path = item_output_dir / UNCALIBRATED_OUTPUT_FILENAME
        if uncalibrated_output_path.is_file():
            uncalibrated_output = ConfEstimationOutput.model_validate_json(uncalibrated_output_path.read_text())
        else:
            saved_output = self.estimator.get_saved_output(item_output_dir)
            if saved_output is None:
                return None
            uncalibrated_output = saved_output
            self._save_uncalibrated_output(uncalibrated_output, item_output_dir)

        calibrated_output = self._calibrate_output(uncalibrated_output)
        self.save_output(calibrated_output, item_output_dir)
        return calibrated_output
