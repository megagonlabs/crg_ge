import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_current_log_path: ContextVar[Path | None] = ContextVar("current_log_path", default=None)


class _ProblemLogFilter(logging.Filter):
    def __init__(self, log_path: Path) -> None:
        super().__init__()
        self.log_path = log_path

    def filter(self, record: logging.LogRecord) -> bool:
        return _current_log_path.get() == self.log_path


@contextmanager
def problem_log_context(log_path: Path) -> Iterator[None]:
    """Attach a file handler that only accepts records from this problem context."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.addFilter(_ProblemLogFilter(log_path))
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(process)d %(threadName)s %(levelname)s %(name)s:%(lineno)d %(message)s")
    )

    root_logger = logging.getLogger()
    token = _current_log_path.set(log_path)
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()
        _current_log_path.reset(token)
