import argparse
import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml

logger = logging.getLogger(__name__)

Operation = Literal["copy", "move"]
_RUN_CONFIG_SUFFIXES = {".yaml", ".yml"}
_EXCLUDED_OUTPUT_ARTIFACTS = {"instantiated_run_config.yaml"}


def _default_output_dir(run_config_path: Path) -> Path:
    run_path = run_config_path.with_suffix("")
    if run_path.is_absolute():
        run_path = Path(*run_path.parts[1:])
    return Path("outputs") / run_path


def _destination_config_path(source_config_path: Path, destination: Path) -> Path:
    if destination.suffix in _RUN_CONFIG_SUFFIXES:
        return destination
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Experiment destination must be a directory or YAML path: {destination}")
    return destination / source_config_path.name


def _require_inferred_output_dir(source_config_path: Path) -> None:
    config_data = yaml.safe_load(source_config_path.read_text())
    if not isinstance(config_data, dict):
        raise TypeError(f"Expected YAML object in {source_config_path}")
    output_config = config_data.get("output")
    if output_config is not None and not isinstance(output_config, dict):
        raise TypeError(f"Expected output object in {source_config_path}")
    if output_config and output_config.get("output_dir") is not None:
        raise ValueError(
            f"{source_config_path} has an explicit output.output_dir; "
            "move it manually because it is not derived from the run-config path"
        )


def transfer_experiment(
    source_config_path: Path,
    destination: Path,
    *,
    operation: Operation,
    overwrite: bool = False,
) -> None:
    """Copy or move an inferred-output experiment, omitting unused instantiated configuration metadata."""
    if source_config_path.suffix not in _RUN_CONFIG_SUFFIXES:
        raise ValueError(f"Run config must be a YAML file: {source_config_path}")
    if not source_config_path.is_file():
        raise FileNotFoundError(f"Run config does not exist: {source_config_path}")
    _require_inferred_output_dir(source_config_path)

    destination_config_path = _destination_config_path(source_config_path, destination)
    if source_config_path.resolve() == destination_config_path.resolve():
        raise ValueError("Experiment destination is the source run config")
    if overwrite and operation != "copy":
        raise ValueError("--overwrite is supported only when copying experiments")
    if destination_config_path.exists() and not overwrite:
        raise FileExistsError(f"Destination run config already exists: {destination_config_path}")

    source_output_dir = _default_output_dir(source_config_path)
    destination_output_dir = _default_output_dir(destination_config_path)
    if destination_output_dir.exists() and not overwrite:
        raise FileExistsError(f"Destination output directory already exists: {destination_output_dir}")
    if source_output_dir.exists() and not source_output_dir.is_dir():
        raise NotADirectoryError(f"Expected experiment output directory: {source_output_dir}")

    if overwrite:
        destination_config_path.unlink(missing_ok=True)
        shutil.rmtree(destination_output_dir, ignore_errors=True)

    transfer_file: Callable[[str | Path, str | Path], str | Path]
    if operation == "move":
        transfer_file = shutil.move
    elif operation == "copy":
        transfer_file = shutil.copy2
    else:
        raise ValueError(f"Unsupported experiment transfer operation: {operation}")

    destination_config_path.parent.mkdir(parents=True, exist_ok=True)
    transfer_file(source_config_path, destination_config_path)
    logger.info("%s run config from %s to %s", operation.capitalize(), source_config_path, destination_config_path)

    if not source_output_dir.exists():
        return

    destination_output_dir.parent.mkdir(parents=True, exist_ok=True)
    if operation == "move":
        shutil.move(source_output_dir, destination_output_dir)
        for artifact_name in _EXCLUDED_OUTPUT_ARTIFACTS:
            (destination_output_dir / artifact_name).unlink(missing_ok=True)
    else:
        shutil.copytree(
            source_output_dir,
            destination_output_dir,
            ignore=shutil.ignore_patterns(*_EXCLUDED_OUTPUT_ARTIFACTS),
        )
    logger.info("%s results from %s to %s", operation.capitalize(), source_output_dir, destination_output_dir)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy or move a run config and its inferred output directory.")
    parser.add_argument("run_config", type=Path, help="Path to the run-config YAML file")
    parser.add_argument("destination", type=Path, help="Destination directory or YAML file path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the destination run config and its inferred output directory before copying (copy_exp only)",
    )
    return parser.parse_args(argv)


def _cli(operation: Operation) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    transfer_experiment(args.run_config, args.destination, operation=operation, overwrite=args.overwrite)


def move_cli() -> None:
    _cli("move")


def copy_cli() -> None:
    _cli("copy")
