import pytest

from crg_ce.estimators.base_estimator import rescale_confidence


def test_rescale_confidence() -> None:
    assert rescale_confidence(min_score=0, max_score=10, score=10) == 1
    assert rescale_confidence(min_score=0, max_score=100, score=10) == 0.1
    assert rescale_confidence(min_score=1, max_score=101, score=11) == 0.1
    assert rescale_confidence(min_score=1, max_score=5, score=3) == 0.5


def test_rescale_confidence_rejects_out_of_scale_scores() -> None:
    with pytest.raises(ValueError, match="between 0 and 10: 75"):
        rescale_confidence(min_score=0, max_score=10, score=75)
