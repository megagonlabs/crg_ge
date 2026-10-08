import logging
from pathlib import Path

from crg_ce.utils.problem_logging import problem_log_context


def test_problem_log_context_filters_records_by_active_problem(tmp_path: Path) -> None:
    # This verifies nested problem log handlers only accept records for their active problem context.
    first_log_path = tmp_path / "first.log"
    second_log_path = tmp_path / "second.log"
    logger = logging.getLogger("crg_ce.tests.problem_logging")

    with problem_log_context(first_log_path):
        logger.warning("first marker")
        with problem_log_context(second_log_path):
            logger.warning("second marker")
        logger.warning("first marker again")

    first_log_text = first_log_path.read_text()
    second_log_text = second_log_path.read_text()

    assert "first marker" in first_log_text
    assert "first marker again" in first_log_text
    assert "second marker" not in first_log_text
    assert "second marker" in second_log_text
    assert "first marker" not in second_log_text
