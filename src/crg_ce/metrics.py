from typing import Annotated

import numpy as np
import pandas as pd
from pydantic import AfterValidator, BaseModel, Field
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def _validate_prob_or_nan(value: float) -> float:
    if np.isnan(value):
        return value
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("must be between 0.0 and 1.0 or NaN")
    return float(value)


ZeroToOneOrNan = Annotated[float, AfterValidator(_validate_prob_or_nan)]


def auroc(*, confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Area under ROC curve. Higher = confidence ranks correct higher than wrong."""
    outcomes = np.asarray(correctness, dtype=int)
    if outcomes.size == 0 or outcomes.min() == outcomes.max():
        return float("nan")
    return float(roc_auc_score(outcomes, confidence))


def auarc(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Area under Accuracy-Rejection Curve (sweep coverage 0→1).

    Higher = model abstains on the wrong cases preferentially.
    """
    scores = np.asarray(confidence)
    outcomes = np.asarray(correctness, dtype=int)
    count = scores.size
    if count == 0:
        raise ValueError("AUARC requires at least one example")
    if count == 1:
        return 0.0
    # Original input order breaks confidence ties, preserving the evaluation convention.
    ranking = np.lexsort((np.arange(count), -scores))
    accuracies = np.add.accumulate(outcomes[ranking]) / np.arange(1, count + 1)
    segment_heights = accuracies[:-1] + accuracies[1:]
    return float(segment_heights.sum() / (2 * count))


def selective_utility(
    confidence: np.ndarray,
    correctness: np.ndarray,
    threshold: float,
) -> float:
    """Return fixed-threshold BAS utility, averaged over examples."""
    if not 0 <= threshold < 1:
        raise ValueError("threshold must lie in [0, 1).")

    answered = confidence >= threshold
    utility = correctness - (1 - correctness) * threshold / (1 - threshold)
    return float(np.mean(answered * utility))


def beh_align_score(confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Return mean selective utility integrated over cost thresholds from zero to one."""
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(int)
    utility = confidence.copy()  # score = s
    incorrect = correctness == 0
    incorrect_confidence = np.clip(confidence[incorrect], None, 1 - np.finfo(float).eps)
    utility[incorrect] += np.log(1 - incorrect_confidence)  # for incorrect: score  = s + ln(1-s)
    return float(np.mean(utility))


# --------------------------------------------------------------------------
#  Calibration (confidence value matches actual frequency of correctness)
# --------------------------------------------------------------------------


def expected_cal_error(
    *,
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins.

    Lower = confidence-frequency match better. ECE in [0, 1].
    """
    scores = np.asarray(confidence)
    outcomes = np.asarray(correctness, dtype=float)
    if scores.size == 0:
        return float("nan")
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    valid = (scores >= 0) & (scores <= 1)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confidence_sums, _ = np.histogram(scores[valid], bins=edges, weights=scores[valid])
    outcome_sums, _ = np.histogram(scores[valid], bins=edges, weights=outcomes[valid])
    # Count-weighted mean gaps equal absolute bin-sum differences divided by total examples.
    return float(np.abs(outcome_sums - confidence_sums).sum() / scores.size)


def adaptive_expected_cal_error(
    *,
    confidence: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error over equal-count confidence bins.

    Lower is better. Samples are sorted by confidence and divided into bins whose
    sizes differ by at most one; each bin's calibration gap is weighted by its size.
    """
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    confidence = np.asarray(confidence)
    correctness = np.asarray(correctness).astype(float)
    if len(confidence) != len(correctness):
        raise ValueError("confidence and correctness must have equal lengths")
    if len(confidence) == 0:
        return float("nan")

    sorted_indices = np.argsort(confidence, stable=True)
    total_error = 0.0
    for bin_indices in np.array_split(sorted_indices, n_bins):
        if len(bin_indices) == 0:
            continue
        calibration_gap = abs(confidence[bin_indices].mean() - correctness[bin_indices].mean())
        total_error += (len(bin_indices) / len(confidence)) * calibration_gap
    return float(total_error)

def brier_score(*, confidence: np.ndarray, correctness: np.ndarray) -> float:
    return float(brier_score_loss(y_true=correctness, y_proba=confidence))  # pyright: ignore[reportCallIssue]


def brier_skill_score(*, confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Brier improvement over the constant empirical base-rate forecast; higher is better."""
    correctness = np.asarray(correctness)
    base_rate = correctness.mean()
    reference_brier = brier_score(confidence=np.full(correctness.shape, base_rate), correctness=correctness)
    if reference_brier == 0:
        return float("nan")
    return 1 - brier_score(confidence=confidence, correctness=correctness) / reference_brier


def negative_log_likelihood(*, confidence: np.ndarray, correctness: np.ndarray) -> float:
    """Mean binary log loss, with probabilities clipped to sklearn's machine-precision epsilon."""
    return float(log_loss(y_true=correctness, y_proba=confidence, labels=[0, 1]))  # pyright: ignore[reportCallIssue]


class ConfMetrics(BaseModel):
    auroc: ZeroToOneOrNan = Field(
        description="Area under the ROC curve for confidence scores against binary correctness; higher is better.",
    )
    n: int = Field(ge=0, description="number of results represented in these metrics")
    ece_10: float = Field(
        ge=0.0,
        le=1.0,
        description="Expected calibration error for confidence scores; lower is better. (10-bins)",
    )
    adaptive_ece_10: float = Field(
        ge=0.0,
        le=1.0,
        description="Adaptive expected calibration error using 10 equal-count bins; lower is better.",
    )
    auarc: float = Field(
        ge=0.0,
        le=1.0,
        description="Area under the accuracy-rejection curve; higher means wrong cases are rejected earlier.",
    )
    beh_align_score: float = Field(
        description="Mean selective utility integrated over all cost thresholds; higher is better.",
    )
    brier_score: float = Field(
        ge=0.0,
        description="Brier Score (mean squared-error of confidence P_pred(success) and correctness P_emp(success))",
    )
    brier_skill_score: float = Field(
        description="Brier improvement over a constant empirical base-rate forecast; 1 is perfect and 0 is no skill.",
    )
    nll: float = Field(
        ge=0.0,
        description="Mean binary negative log-likelihood (log loss); lower is better.",
    )
    mean_est_confidence: float = Field(ge=0.0, le=1.0, description="Mean estimated confidence (informational only)")
    std_est_confidence: float = Field(description="Standard deviation estimated confidence (informational only)")
    mean_accuracy: float = Field(ge=0.0, le=1.0, description="Mean true correctness in this sample")

    def get_metric_log(self) -> str:
        # Programmatically return all model fields in definition order as `name=value` pairs
        parts = []
        for fname in ConfMetrics.model_fields.keys():
            parts.append(f"{fname}={getattr(self, fname)}")
        return ", ".join(parts)


def metrics_from_df(df: pd.DataFrame) -> ConfMetrics:
    confidence: np.ndarray = df["estimated_confidence"].to_numpy()
    correctness: np.ndarray = df["resolved"].to_numpy().astype(np.int64)
    assert not df[["instance_id", "model"]].duplicated().any()
    return ConfMetrics(
        n=len(df),
        ece_10=expected_cal_error(confidence=confidence, correctness=correctness, n_bins=10),
        adaptive_ece_10=adaptive_expected_cal_error(
            confidence=confidence,
            correctness=correctness,
            n_bins=10,
        ),
        auroc=auroc(confidence=confidence, correctness=correctness),
        auarc=auarc(confidence=confidence, correctness=correctness),
        beh_align_score=beh_align_score(confidence=confidence, correctness=correctness),
        brier_score=brier_score(confidence=confidence, correctness=correctness),
        brier_skill_score=brier_skill_score(confidence=confidence, correctness=correctness),
        nll=negative_log_likelihood(confidence=confidence, correctness=correctness),
        mean_accuracy=correctness.mean(),
        mean_est_confidence=confidence.mean(),
        std_est_confidence=confidence.std(),
    )
