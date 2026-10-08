from pathlib import Path

import pandas as pd
import pytest

from crg_ce.experiments.parse_results import build_results_table, main, read_average_usage, read_estimator_model_name


@pytest.mark.parametrize("estimator_type", ["gsn_product", "gsn_aggregate"])
def test_read_estimator_model_name_supports_gsn_aggregation_names(tmp_path: Path, estimator_type: str) -> None:
    # This verifies current and legacy GSN aggregation run configs are reported as having no estimator model.
    (tmp_path / "run_config.yaml").write_text(
        f"""estimator:
  estimator_type: {estimator_type}
  replay_from: runs/source-gsn.yaml
  aggregation_type: product
"""
    )

    assert read_estimator_model_name(tmp_path) == "N/A"


def test_read_average_usage_estimates_cost_when_recorded_cost_is_zero() -> None:
    # This guarantees unavailable zero-cost provider reports fall back to configured token pricing.
    usage = read_average_usage(
        pd.DataFrame({"total_tokens": [10_000], "generated_tokens": [2_000], "cost": [0.0]}),
        Path("results.csv"),
        "openai/Qwen/Qwen3.8-27B-FP8",
    )

    assert usage["avg_cost"] == "$0.01"


def test_by_benchmark_prints_one_table_for_each_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # This verifies per-benchmark reporting recomputes and prints independently filtered metrics for each benchmark.
    benchmarks = ["enterprise-ops-gym", "skillsbench", "swe-smith"]
    for run_name, confidences in {"run-a": [0.2, 0.4, 0.6], "run-b": [0.3, 0.5, 0.7]}.items():
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("{}")
        (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
        pd.DataFrame(
            {
                "instance_id": [f"{benchmark}-instance" for benchmark in benchmarks],
                "model": ["evaluated-model"] * len(benchmarks),
                "benchmark": benchmarks,
                "estimated_confidence": confidences,
                "resolved": [0, 1, 1],
            }
        ).to_csv(run_dir / "results.csv", index=False)

    main(["--base_dir", str(tmp_path), "--by-benchmark"])

    output = capsys.readouterr().out
    assert [line for line in output.splitlines() if line.startswith("Benchmark:")] == [
        "Benchmark: enterprise-ops-gym",
        "Benchmark: skillsbench",
        "Benchmark: swe-smith",
    ]
    assert output.count("estimator_model") == 3


def test_by_model_prints_one_table_for_each_agent_model(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # This verifies per-model reporting recomputes metrics from tasks for each dataset agent model.
    for run_name, confidences in {"run-a": [0.2, 0.8], "run-b": [0.3, 0.7]}.items():
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("{}")
        (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
        pd.DataFrame(
            {
                "instance_id": [f"{run_name}-a", f"{run_name}-b"],
                "model": ["model-a", "model-b"],
                "estimated_confidence": confidences,
                "resolved": [0, 1],
            }
        ).to_csv(run_dir / "results.csv", index=False)

    main(["--base_dir", str(tmp_path), "--by-model"])

    output = capsys.readouterr().out
    assert [line for line in output.splitlines() if line.startswith("Model:")] == [
        "Model: model-a",
        "Model: model-b",
    ]
    assert output.count("| run-a") == 2
    assert output.count("| run-b") == 2


def test_by_difficulty_normalizes_long_tasks_and_includes_none(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # This verifies difficulty reporting combines long tasks and treats missing difficulty as its own group.
    for run_name in ["run-a", "run-b"]:
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("{}")
        (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
        pd.DataFrame(
            {
                "instance_id": [f"{run_name}-{index}" for index in range(4)],
                "model": ["evaluated-model"] * 4,
                "difficulty": ["<15 minutes", "1-4 hours", ">4 hours", None],
                "estimated_confidence": [0.2, 0.4, 0.6, 0.8],
                "resolved": [0, 0, 1, 1],
            }
        ).to_csv(run_dir / "results.csv", index=False)

    main(["--base_dir", str(tmp_path), "--by-difficulty"])

    output = capsys.readouterr().out
    assert [line for line in output.splitlines() if line.startswith("Difficulty:")] == [
        "Difficulty: <15 minutes",
        "Difficulty: > 1 hour",
        "Difficulty: None",
    ]
    assert output.count("| run-a") == 3
    assert output.count("| run-b") == 3
    assert "| run-a | N/A               |   2 |" in output


def test_min_n_excludes_runs_with_too_few_results(tmp_path: Path) -> None:
    # This guarantees --min-n uses the complete run result count and excludes undersized runs.
    for run_name, confidences in {"complete": [0.2, 0.8], "partial": [0.4]}.items():
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text("{}")
        (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
        pd.DataFrame(
            {
                "instance_id": [f"instance-{index}" for index in range(len(confidences))],
                "model": ["evaluated-model"] * len(confidences),
                "estimated_confidence": confidences,
                "resolved": [0, 1][: len(confidences)],
            }
        ).to_csv(run_dir / "results.csv", index=False)

    table = build_results_table(tmp_path, shared_subset=True, min_n=2)

    assert table["run"].tolist() == ["complete"]


def test_min_n_applies_before_benchmark_filtering(tmp_path: Path) -> None:
    # This guarantees a run with enough total results remains in a benchmark table even when that benchmark is smaller.
    run_dir = tmp_path / "complete"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    pd.DataFrame(
        {
            "instance_id": ["skills-instance", "swe-instance"],
            "model": ["evaluated-model", "evaluated-model"],
            "benchmark": ["skillsbench", "swe-smith"],
            "estimated_confidence": [0.2, 0.8],
            "resolved": [0, 1],
        }
    ).to_csv(run_dir / "results.csv", index=False)

    table = build_results_table(tmp_path, benchmark="skillsbench", min_n=2)

    assert table["run"].tolist() == ["complete"]
    assert table["n"].tolist() == [1]


def test_tsv_writes_a_spreadsheet_pasteable_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # This verifies --tsv writes the normal result table as tab-separated rows to standard output.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    pd.DataFrame(
        {
            "instance_id": ["instance"],
            "model": ["evaluated-model"],
            "estimated_confidence": [0.5],
            "resolved": [1],
        }
    ).to_csv(run_dir / "results.csv", index=False)

    main(["--base_dir", str(tmp_path), "--tsv"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("run\testimator_model\t")
    assert lines[1].startswith("run\tN/A\t")


def test_simple_tsv_includes_only_requested_metric_columns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # This verifies --simple omits model and usage summaries while preserving the requested compact metric table.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        '{"n": 2, "ece_10": 0.1, "adaptive_ece_10": 0.15, '
        '"brier_score": 0.2, "brier_skill_score": 0.3, '
        '"auroc": 0.7, "auarc": 0.8, '
        '"beh_align_score": 0.9, "mean_est_confidence": 0.6}'
    )
    (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    pd.DataFrame(
        {
            "instance_id": ["first", "second"],
            "model": ["evaluated-model", "evaluated-model"],
            "estimated_confidence": [0.5, 0.7],
            "resolved": [0, 1],
        }
    ).to_csv(run_dir / "results.csv", index=False)

    main(["--base_dir", str(tmp_path), "--tsv", "--simple"])

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "run\tn\tece_10 (↓)\tadaptive_ece_10 (↓)\tbrier (↓)\tbrier_skill (↑)\t"
        "auroc (↑)\tauarc (↑)\tbeh_align (↑)\tavg_conf"
    )
    assert lines[1] == "run\t2\t0.10\t0.15\t0.20\t0.30\t0.70\t0.80\t0.90\t0.60"


def test_tsv_by_benchmark_prints_one_pasteable_table_per_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # This verifies --tsv and --by-benchmark produce labeled, separate tab-separated tables for copying.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text("{}")
    (run_dir / "run_config.yaml").write_text("estimator:\n  estimator_type: gsn_aggregate\n")
    pd.DataFrame(
        {
            "instance_id": ["skills-instance", "swe-instance"],
            "model": ["evaluated-model", "evaluated-model"],
            "benchmark": ["skillsbench", "swe-smith"],
            "estimated_confidence": [0.5, 0.5],
            "resolved": [1, 1],
        }
    ).to_csv(run_dir / "results.csv", index=False)

    main(["--base_dir", str(tmp_path), "--tsv", "--by-benchmark"])

    output = capsys.readouterr().out
    assert "Benchmark: skillsbench\nrun\testimator_model\t" in output
    assert "Benchmark: swe-smith\nrun\testimator_model\t" in output
