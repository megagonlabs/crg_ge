from pathlib import Path

import pandas as pd
import pytest

from crg_ce.experiments.ece_sweep import build_ece_sweep_table, main
from crg_ce.metrics import adaptive_expected_cal_error, expected_cal_error


def _write_run(base_dir: Path, run_name: str, confidences: list[float], correctness: list[int]) -> None:
    run_dir = base_dir / run_name
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    pd.DataFrame(
        {
            "instance_id": [f"instance-{index}" for index in range(len(confidences))],
            "model": ["evaluated-model"] * len(confidences),
            "estimated_confidence": confidences,
            "resolved": correctness,
        }
    ).to_csv(run_dir / "results.csv", index=False)


def test_build_ece_sweep_table_reports_each_requested_bin_count(tmp_path: Path) -> None:
    # This verifies the sweep recomputes equal-width and adaptive ECE for every requested bin count.
    confidences = [0.1, 0.4, 0.6, 0.9]
    correctness = [0, 1, 0, 1]
    _write_run(tmp_path, "run", confidences, correctness)

    table = build_ece_sweep_table(tmp_path, bins=[2, 4])

    assert table.columns.tolist() == [
        "run",
        "estimator_model",
        "n",
        "ece_2 (↓)",
        "ece_4 (↓)",
        "adaptive_ece_2 (↓)",
        "adaptive_ece_4 (↓)",
    ]
    assert table.loc[0, "ece_2 (↓)"] == pytest.approx(
        expected_cal_error(confidence=confidences, correctness=correctness, n_bins=2)
    )
    assert table.loc[0, "adaptive_ece_4 (↓)"] == pytest.approx(
        adaptive_expected_cal_error(confidence=confidences, correctness=correctness, n_bins=4)
    )


def test_main_emits_simple_tsv_for_custom_bins(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # This verifies the CLI accepts custom bins and emits a spreadsheet-pasteable compact table.
    _write_run(tmp_path, "run", [0.2, 0.8], [0, 1])

    main(["--base_dir", str(tmp_path), "--bins", "2", "3", "--simple", "--tsv"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "run\tn\tece_2 (↓)\tece_3 (↓)\tadaptive_ece_2 (↓)\tadaptive_ece_3 (↓)"
    assert lines[1].startswith("run\t2\t")


@pytest.mark.parametrize("bins", [[0], [-1], [5, 5]])
def test_build_ece_sweep_table_rejects_invalid_bins(tmp_path: Path, bins: list[int]) -> None:
    # This guarantees invalid sweep definitions fail explicitly before reading experiment artifacts.
    with pytest.raises(ValueError):
        build_ece_sweep_table(tmp_path, bins=bins)
