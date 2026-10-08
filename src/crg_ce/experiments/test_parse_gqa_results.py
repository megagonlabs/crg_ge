import json
from pathlib import Path

import pandas as pd
import pytest

from crg_ce.experiments.parse_gqa_results import build_results_table


def test_build_results_table_sums_costs_and_reports_failures(tmp_path: Path) -> None:
    # This verifies the GQA summary retains aggregate scores while computing total usage from per-graph records.
    # It assumes metrics.json and results.csv are internally consistent local-GQA outputs.
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "n": 2,
                "joint_sufficiency": {"score": 0.5},
                "child_necessity": {"score": 0.75},
                "non_redundant_siblings": None,
                "quality_of_particularization": None,
                "total_entailments": {"score": 0.6, "observed_count": 3, "expected_count": 5},
                "avg_total_tokens": 15.0,
                "avg_generated_tokens": 4.0,
                "avg_cost": 0.15,
            }
        )
    )
    pd.DataFrame(
        {
            "evaluator_model": ["test-model", "test-model"],
            "total_tokens": [10, 20],
            "generated_tokens": [3, 5],
            "cost": [0.1, 0.2],
        }
    ).to_csv(run_dir / "results.csv", index=False)
    (run_dir / "failures.json").write_text(json.dumps([{"error": "test"}]))

    results = build_results_table(tmp_path)

    assert results.loc[0, "run"] == "run-a"
    assert results.loc[0, "failed"] == 1
    assert results.loc[0, "total_entailments"] == 0.6
    assert results.loc[0, "total_cost"] == pytest.approx(0.3)
    assert results.loc[0, "total_tokens"] == 30
