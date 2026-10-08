"""Optional on-disk logging for prompts sent to LLMs."""

import os
from pathlib import Path


def log_prompt(prompt: str, output_dir: Path) -> None:
    """Write a prompt to the next numbered file when ``LOG_PROMPTS`` is truthy."""
    if os.environ.get("LOG_PROMPTS", "").lower() not in {"1", "true", "yes", "on"}:
        return

    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompt_index = 0
    while True:
        prompt_path = prompts_dir / f"{prompt_index}.txt"
        try:
            file_descriptor = os.open(prompt_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            prompt_index += 1
            continue
        with open(file_descriptor, "w") as prompt_file:
            prompt_file.write(prompt)
        return
