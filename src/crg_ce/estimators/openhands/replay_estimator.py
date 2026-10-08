"""Reuse saved final confidences for calibration or hyperparameter runs without repeating model calls."""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from crg_ce.estimators.base_estimator import (
    BaseConfidenceEstimator,
    ConfEstimationInput,
    ConfEstimationOutput,
    output_path_for_item,
)
from crg_ce.estimators.calibration import CalibrationConfig


class ReplayEstimatorConfig(BaseModel):
    """Configure an estimator that reuses outputs from a completed confidence-estimation run."""

    estimator_type: Literal["replay"] = "replay"
    replay_from: Path = Field(
        description="Run-config YAML whose output.json artifacts provide the uncalibrated confidence estimates."
    )
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class ReplayConfidenceEstimator(BaseConfidenceEstimator):
    """Return a saved confidence estimate for the current dataset item from another run."""

    def __init__(self, cfg: ReplayEstimatorConfig) -> None:
        from crg_ce.estimators.openhands.config import output_dir_for_run_config

        self.replay_output_dir = output_dir_for_run_config(cfg.replay_from)

    def estimate_confidence(self, ce_input: ConfEstimationInput) -> ConfEstimationOutput:
        source_output_path = output_path_for_item(self.replay_output_dir, ce_input.instance_id, ce_input.model)
        if not source_output_path.is_file():
            raise FileNotFoundError(
                f"Replay output does not exist for instance_id={ce_input.instance_id!r}, "
                f"model={ce_input.model!r}: {source_output_path}"
            )
        output = ConfEstimationOutput.model_validate_json(source_output_path.read_text())
        self.save_output(output, ce_input.output_dir)
        return output
