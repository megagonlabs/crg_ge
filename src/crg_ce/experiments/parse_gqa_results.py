"""Summarize local GQA run outputs."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_BASE_DIR = Path("outputs/runs/test_set/gqa/local")
DEFAULT_OUTPUT_FILENAME = "gqa_results_summary.csv"
FLOAT_FORMAT = ".3f"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize local GQA run outputs.")
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_path", type=Path, default=None)
    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument("--csv", action="store_true", help="Write CSV to stdout instead of saving a summary")
    output_format.add_argument("--tsv", action="store_true", help="Write TSV to stdout instead of saving a summary")
    parser.add_argument("--min-n", type=int, default=None, help="Exclude runs with fewer than N completed graphs")
    return parser.parse_args(argv)


def _required_metrics(metrics_path: Path) -> dict[str, Any]:
    metrics = json.loads(metrics_path.read_text())
    if not isinstance(metrics, dict):
        raise TypeError(f"Expected metrics object in {metrics_path}, got {type(metrics).__name__}")
    required_keys = {"n", "avg_total_tokens", "avg_generated_tokens", "avg_cost", "total_entailments"}
    missing_keys = required_keys - metrics.keys()
    if missing_keys:
        raise ValueError(f"Missing local GQA metrics in {metrics_path}: {sorted(missing_keys)}")
    return metrics


def _metric_score(metrics: dict[str, Any], metric_name: str) -> float | None:
    metric = metrics[metric_name]
    if metric is None:
        return None
    if not isinstance(metric, dict) or "score" not in metric:
        raise ValueError(f"Invalid {metric_name} metric")
    return metric["score"]  # type: ignore


def build_results_table(base_dir: Path, *, min_n: int | None = None) -> pd.DataFrame:
    if min_n is not None and min_n < 1:
        raise ValueError("--min-n must be at least 1")
    metrics_paths = sorted(base_dir.rglob("metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics.json files found under {base_dir}")

    rows = []
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        metrics = _required_metrics(metrics_path)
        n = metrics["n"]
        if not isinstance(n, int) or n < 0:
            raise ValueError(f"Invalid n in {metrics_path}")
        if min_n is not None and n < min_n:
            continue

        results_path = run_dir / "results.csv"
        results = pd.read_csv(results_path)
        if len(results) != n:
            raise ValueError(f"metrics n={n} does not match {len(results)} rows in {results_path}")
        if "evaluator_model" not in results.columns or "cost" not in results.columns:
            raise ValueError(f"Missing evaluator_model or cost column in {results_path}")
        evaluator_models = results["evaluator_model"].dropna().unique()
        if len(evaluator_models) != 1:
            raise ValueError(f"Expected one evaluator model in {results_path}, got {list(evaluator_models)}")
        total_cost = float(pd.to_numeric(results["cost"], errors="raise").sum())
        total_tokens = int(pd.to_numeric(results["total_tokens"], errors="raise").sum())
        total_generated_tokens = int(pd.to_numeric(results["generated_tokens"], errors="raise").sum())
        total_entailments = metrics["total_entailments"]
        if not isinstance(total_entailments, dict):
            raise ValueError(f"Invalid total_entailments metric in {metrics_path}")

        failures_path = run_dir / "failures.json"
        failures = json.loads(failures_path.read_text())
        if not isinstance(failures, list):
            raise TypeError(f"Expected failures list in {failures_path}, got {type(failures).__name__}")
        rows.append(
            {
                "run": run_dir.relative_to(base_dir).as_posix(),
                "evaluator_model": evaluator_models[0],
                "n": n,
                "failed": len(failures),
                "joint_sufficiency": _metric_score(metrics, "joint_sufficiency"),
                "child_necessity": _metric_score(metrics, "child_necessity"),
                "non_redundant_siblings": _metric_score(metrics, "non_redundant_siblings"),
                "quality_of_particularization": _metric_score(metrics, "quality_of_particularization"),
                "total_entailments": total_entailments["score"],
                "observed_entailments": total_entailments["observed_count"],
                "expected_entailments": total_entailments["expected_count"],
                "avg_total_tokens": metrics["avg_total_tokens"],
                "total_tokens": total_tokens,
                "avg_generated_tokens": metrics["avg_generated_tokens"],
                "total_generated_tokens": total_generated_tokens,
                "avg_cost": metrics["avg_cost"],
                "total_cost": total_cost,
            }
        )
    if not rows:
        raise ValueError(f"No runs under {base_dir} have at least {min_n} completed graphs")
    return pd.DataFrame(rows).sort_values("run")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    results = build_results_table(args.base_dir, min_n=args.min_n)
    if args.csv:
        print(results.to_csv(index=False, float_format=f"%{FLOAT_FORMAT}"), end="")
        return
    if args.tsv:
        print(results.to_csv(index=False, sep="\t", float_format=f"%{FLOAT_FORMAT}"), end="")
        return

    output_path = args.output_path or args.base_dir / DEFAULT_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, float_format=f"%{FLOAT_FORMAT}")
    print(f"Saved GQA results to {output_path}")
    print(results.to_markdown(index=False, floatfmt=FLOAT_FORMAT))


def cli() -> None:
    main()


if __name__ == "__main__":
    cli()
