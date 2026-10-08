from pathlib import Path

from crg_ce.utils.prompt_logging import log_prompt


def test_log_prompt_writes_incrementing_prompt_files(tmp_path: Path, monkeypatch) -> None:
    # This verifies enabled prompt logging preserves prompt order without overwriting prior requests.
    monkeypatch.setenv("LOG_PROMPTS", "true")
    log_prompt("first prompt", tmp_path)
    log_prompt("second prompt", tmp_path)

    assert (tmp_path / "prompts" / "0.txt").read_text() == "first prompt"
    assert (tmp_path / "prompts" / "1.txt").read_text() == "second prompt"


def test_log_prompt_is_disabled_without_environment_variable(tmp_path: Path) -> None:
    # This verifies prompt logging is opt-in through the process environment.
    log_prompt("unlogged prompt", tmp_path)
    assert not (tmp_path / "prompts").exists()
