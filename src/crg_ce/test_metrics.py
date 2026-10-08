import numpy as np
import pandas as pd
import pytest

from crg_ce.metrics import (
    adaptive_expected_cal_error,
    auarc,
    auroc,
    beh_align_score,
    brier_skill_score,
    expected_cal_error,
    metrics_from_df,
    negative_log_likelihood,
)


def test_beh_align_score_integrates_selective_utility_per_outcome() -> None:
    # This verifies correct and incorrect results use the closed forms obtained by integrating selective utility.
    score = beh_align_score(np.array([0.5, 0.5]), np.array([1, 0]))

    assert score == (0.5 + 0.5 + np.log(0.5)) / 2


def test_metrics_from_df_includes_beh_align_score() -> None:
    # This verifies every standard metrics computation, including partial metric generation, persists the new score.
    metrics = metrics_from_df(
        pd.DataFrame(
            {
                "instance_id": ["first", "second"],
                "model": ["model", "model"],
                "estimated_confidence": [0.5, 0.5],
                "resolved": [1, 0],
            }
        )
    )

    assert metrics.beh_align_score == (0.5 + 0.5 + np.log(0.5)) / 2


def test_negative_log_likelihood_is_included_in_standard_metrics() -> None:
    # This verifies standard metrics report mean binary log loss using confidence as P(resolved).
    confidence = np.array([0.8, 0.2])
    correctness = np.array([1, 0])
    metrics = metrics_from_df(
        pd.DataFrame(
            {
                "instance_id": ["first", "second"],
                "model": ["model", "model"],
                "estimated_confidence": confidence,
                "resolved": correctness,
            }
        )
    )

    expected = -np.log(0.8)
    assert negative_log_likelihood(confidence=confidence, correctness=correctness) == expected
    assert metrics.nll == expected
    assert metrics.brier_skill_score == pytest.approx(0.84)


def test_brier_skill_score_is_undefined_for_a_constant_outcome() -> None:
    # This verifies skill is NaN when the empirical base-rate forecast has zero Brier loss.
    score = brier_skill_score(confidence=np.array([1.0, 1.0]), correctness=np.array([1, 1]))

    assert np.isnan(score)


def test_adaptive_expected_cal_error_uses_equal_count_bins() -> None:
    # This verifies adaptive ECE sorts examples into equally sized bins rather than fixed-width confidence ranges.
    confidence = np.array([0.1, 0.2, 0.3, 0.9])
    correctness = np.array([0, 1, 0, 1])

    score = adaptive_expected_cal_error(confidence=confidence, correctness=correctness, n_bins=2)

    assert score == pytest.approx(0.225)


def test_metrics_from_df_includes_ten_bin_adaptive_ece() -> None:
    # This guarantees persisted standard metrics expose adaptive ECE with its default ten equal-count bins.
    metrics = metrics_from_df(
        pd.DataFrame(
            {
                "instance_id": ["first", "second"],
                "model": ["model", "model"],
                "estimated_confidence": [0.2, 0.8],
                "resolved": [0, 1],
            }
        )
    )

    assert metrics.adaptive_ece_10 == pytest.approx(0.2)


def test_equal_width_ece_assigns_internal_edges_to_the_next_bin() -> None:
    # This fixes the metric convention at bin boundaries, including the closed final endpoint.
    score = expected_cal_error(
        confidence=np.array([0.0, 0.5, 1.0]), correctness=np.array([1, 0, 1]), n_bins=2
    )

    assert score == pytest.approx(0.5)


def test_auarc_retains_input_order_for_confidence_ties() -> None:
    # This verifies ties preserve the historical ranking convention and trapezoidal endpoint weights.
    confidence = np.array([0.8, 0.8, 0.2])

    assert auarc(confidence, np.array([0, 1, 1])) == pytest.approx(5 / 18)
    assert auarc(confidence, np.array([1, 0, 1])) == pytest.approx(4 / 9)
    assert auarc(np.array([0.8]), np.array([1])) == 0.0


def test_auroc_reports_undefined_single_class_and_correct_ranking() -> None:
    # This ensures delegation to scikit-learn preserves ranking semantics and undefined one-class results.
    assert auroc(confidence=np.array([0.1, 0.9]), correctness=np.array([0, 1])) == 1.0
    assert np.isnan(auroc(confidence=np.array([0.1, 0.9]), correctness=np.array([1, 1])))
