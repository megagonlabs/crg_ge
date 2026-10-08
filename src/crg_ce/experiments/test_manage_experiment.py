from pathlib import Path

import pytest

from crg_ce.experiments.manage_experiment import transfer_experiment


def _create_experiment(root: Path) -> tuple[Path, Path]:
    config_path = root / "runs/exploration/source/demo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("dataset: {}\n")
    output_dir = root / "outputs/runs/exploration/source/demo"
    output_dir.mkdir(parents=True)
    (output_dir / "results.csv").write_text("instance_id,estimated_confidence\nexample,0.5\n")
    (output_dir / "instantiated_run_config.yaml").write_text("unused instantiated config\n")
    return config_path, output_dir


def test_transfer_experiment_moves_config_results_and_omits_instantiated_config(tmp_path: Path, monkeypatch) -> None:
    # This verifies moving a default-layout experiment moves results while dropping unused instantiated metadata.
    monkeypatch.chdir(tmp_path)
    source_config, source_output = _create_experiment(tmp_path)

    transfer_experiment(source_config.relative_to(tmp_path), Path("runs/exploration/destination"), operation="move")

    destination_config = tmp_path / "runs/exploration/destination/demo.yaml"
    destination_output = tmp_path / "outputs/runs/exploration/destination/demo"
    assert not source_config.exists()
    assert not source_output.exists()
    assert destination_config.read_text() == "dataset: {}\n"
    assert (destination_output / "results.csv").exists()
    assert not (destination_output / "instantiated_run_config.yaml").exists()


def test_transfer_experiment_copies_to_an_explicit_config_path(tmp_path: Path, monkeypatch) -> None:
    # This verifies copying preserves the source while deriving results from an explicit destination filename.
    monkeypatch.chdir(tmp_path)
    source_config, source_output = _create_experiment(tmp_path)

    transfer_experiment(source_config.relative_to(tmp_path), Path("runs/archive/renamed.yaml"), operation="copy")

    assert source_config.exists()
    assert source_output.exists()
    assert (tmp_path / "runs/archive/renamed.yaml").exists()
    assert (tmp_path / "outputs/runs/archive/renamed/results.csv").exists()


def test_transfer_experiment_rejects_explicit_output_directory(tmp_path: Path, monkeypatch) -> None:
    # This verifies custom output locations fail explicitly instead of leaving related artifacts behind.
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "runs/source.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("output:\n  output_dir: outputs/custom\n")

    with pytest.raises(ValueError, match="explicit output.output_dir"):
        transfer_experiment(config_path.relative_to(tmp_path), Path("runs/destination"), operation="move")


def test_transfer_experiment_copy_omits_unusable_instantiated_config(tmp_path: Path, monkeypatch) -> None:
    # This verifies copying results omits the serialized configuration metadata without parsing its Python YAML tags.
    monkeypatch.chdir(tmp_path)
    source_config, source_output = _create_experiment(tmp_path)
    (source_output / "instantiated_run_config.yaml").write_text(
        "dataset:\n"
        "  slice: !!python/object/apply:builtins.slice\n"
        "  - null\n"
        "  - 10\n"
        "  - null\n"
        "output:\n"
        f"  output_dir: {source_output.as_posix()}\n"
    )

    transfer_experiment(source_config.relative_to(tmp_path), Path("runs/destination"), operation="copy")

    assert not (tmp_path / "outputs/runs/destination/demo/instantiated_run_config.yaml").exists()


def test_copy_overwrite_replaces_only_matching_config_and_output(tmp_path: Path, monkeypatch) -> None:
    # This verifies overwrite replaces only the matching experiment while preserving sibling destination artifacts.
    monkeypatch.chdir(tmp_path)
    source_config, _ = _create_experiment(tmp_path)
    destination_config_dir = tmp_path / "runs/exploration/destination"
    destination_config_dir.mkdir(parents=True)
    (destination_config_dir / "stale.yaml").write_text("stale config\n")
    (destination_config_dir / "demo.yaml").write_text("old destination config\n")
    destination_output_dir = tmp_path / "outputs/runs/exploration/destination"
    destination_output_dir.mkdir(parents=True)
    (destination_output_dir / "stale.txt").write_text("stale output\n")
    (destination_output_dir / "demo").mkdir()
    (destination_output_dir / "demo/stale.txt").write_text("old destination output\n")

    transfer_experiment(
        source_config.relative_to(tmp_path),
        Path("runs/exploration/destination"),
        operation="copy",
        overwrite=True,
    )

    assert (destination_config_dir / "stale.yaml").exists()
    assert (destination_config_dir / "demo.yaml").read_text() == "dataset: {}\n"
    assert (destination_output_dir / "stale.txt").exists()
    assert not (destination_output_dir / "demo/stale.txt").exists()
    assert (destination_output_dir / "demo/results.csv").exists()
