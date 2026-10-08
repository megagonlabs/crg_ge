import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from crg_ce.metrics import metrics_from_df

DEFAULT_BASE_DIR = Path("outputs/runs/test/batch_run_ce")
DEFAULT_OUTPUT_FILENAME = "results_summary.csv"
FLOAT_FORMAT = ".2f"
METRIC_COLUMN_LABELS = {
    "n": "n",
    "ece_10": "ece_10 (↓)",
    "adaptive_ece_10": "adaptive_ece_10 (↓)",
    "brier_score": "brier (↓)",
    "brier_skill_score": "brier_skill (↑)",
    "auroc": "auroc (↑)",
    "auarc": "auarc (↑)",
    "beh_align_score": "beh_align (↑)",
    "mean_est_confidence": "avg_conf",
    "std_est_confidence": "std_conf",
    "mean_accuracy": "mean_acc",
}
CORE_METRIC_ORDER = [
    "n",
    "ece_10",
    "adaptive_ece_10",
    "brier_score",
    "brier_skill_score",
    "auroc",
    "auarc",
    "beh_align_score",
]
TOKEN_USAGE_COLUMNS = ["total_tokens", "generated_tokens"]
MODEL_COST_PER_MILLION_TOKENS = {
    "Qwen/Qwen3.6-27B": {"input": 0.289, "output": 2.40},
    "openai/qwen36-27b": {"input": 0.289, "output": 2.40},
    "openai/qwen38-27b": {"input": 0.289, "output": 2.40},
    "Qwen/Qwen3.8-27B-FP8": {"input": 0.32, "output": 2.50},
    "openai/Qwen/Qwen3.8-27B-FP8": {"input": 0.32, "output": 2.50},
}
USAGE_SUMMARY_ORDER = ["avg_total_tokens", "avg_generated_tokens", "avg_cost"]
USAGE_COLUMN_LABELS = {
    "avg_total_tokens": "total tokens",
    "avg_generated_tokens": "gen tokens",
    "avg_cost": "cost",
}
SHARED_SUBSET_COLUMNS = ["instance_id", "model"]
SIMPLE_RESULT_COLUMNS = [
    "run",
    "n",
    "ece_10 (↓)",
    "adaptive_ece_10 (↓)",
    "brier (↓)",
    "brier_skill (↑)",
    "auroc (↑)",
    "auarc (↑)",
    "beh_align (↑)",
    "avg_conf",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_path", type=Path, default=None)
    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument(
        "--csv", action="store_true", help="Write CSV to stdout instead of saving a summary file"
    )
    output_format.add_argument(
        "--tsv", action="store_true", help="Write TSV to stdout instead of saving a summary file"
    )
    parser.add_argument(
        "--shared-subset",
        action="store_true",
        help="Recompute results for the instance/model pairs present in every run",
    )
    split_by = parser.add_mutually_exclusive_group()
    split_by.add_argument(
        "--by-benchmark",
        action="store_true",
        help="Print a separate metrics table for each benchmark",
    )
    split_by.add_argument(
        "--by-model",
        action="store_true",
        help="Print a separate metrics table for each agent model",
    )
    split_by.add_argument(
        "--by-difficulty",
        action="store_true",
        help="Print a separate metrics table for each normalized difficulty",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=None,
        help="Exclude runs with fewer than N total evaluated rows before benchmark filtering or a shared subset",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Output only the run name and core calibration/discrimination metrics",
    )
    return parser.parse_args(argv)


def get_metrics_paths(base_dir: Path) -> list[Path]:
    metrics_paths = sorted(base_dir.rglob("metrics.json"))
    if not metrics_paths:
        raise FileNotFoundError(f"No metrics.json files found under {base_dir}")
    return metrics_paths


def read_metrics(metrics_path: Path) -> dict[str, Any]:
    data = json.loads(metrics_path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"Expected metrics object in {metrics_path}, got {type(data).__name__}")
    return data


def read_estimator_model_name(run_dir: Path) -> str:
    config_path = run_dir / "run_config.yaml"
    config_data = yaml.safe_load(config_path.read_text())
    if not isinstance(config_data, dict):
        raise TypeError(f"Expected config object in {config_path}, got {type(config_data).__name__}")

    estimator = config_data["estimator"]
    if not isinstance(estimator, dict):
        raise TypeError(f"Expected estimator object in {config_path}, got {type(estimator).__name__}")

    match estimator["estimator_type"]:
        case "oh_gsn":
            graph_component = estimator.get("graph_populator") or estimator["graph_generator"]
            return graph_component["agent"]["model_name"]  # type: ignore
        case "litellm_verbal":
            return estimator["litellm"]["agent"]["model_name"]  # type: ignore
        case "log_probs_estimator":
            return estimator["agent"]["model_name"]  # type: ignore
        case "dummy":
            return "N/A"
        case "replay":
            return "replay"
        case "gsn_product" | "gsn_aggregate":
            return "N/A"
        case "gsn_plain_verbalized":
            return estimator.get("model_name")  # type: ignore
        case estimator_type:
            raise ValueError(f"Unsupported estimator type in {config_path}: {estimator_type!r}")


def read_average_usage(results: pd.DataFrame, results_path: Path, model: str) -> dict[str, str]:
    missing_columns = set(TOKEN_USAGE_COLUMNS) - set(results.columns)
    if missing_columns:
        return {
            "avg_total_tokens": "-",
            "avg_generated_tokens": "-",
            "avg_cost": "-",
        }

    usage = results[TOKEN_USAGE_COLUMNS].apply(pd.to_numeric, errors="raise")
    usage = usage.fillna(-1)
    if (usage < -1).any().any():
        raise ValueError(f"Invalid usage values below -1 in {results_path}")

    if "cost" in results.columns:
        costs = pd.to_numeric(results["cost"], errors="raise").fillna(-1)
        if (costs < -1).any():
            raise ValueError(f"Invalid cost values below -1 in {results_path}")
    else:
        costs = pd.Series(-1, index=results.index, dtype=float)

    averages = {
        column: values[values != -1].mean() if (values != -1).any() else None for column, values in usage.items()
    }

    def format_tokens(value: float | None) -> str:
        return "N/A" if value is None else f"{value / 1_000:.1f}K"

    average_cost = costs[costs != -1].mean() if (costs != -1).any() else None
    if average_cost in (None, 0.0) and model in MODEL_COST_PER_MILLION_TOKENS:
        average_total_tokens = averages["total_tokens"]
        average_generated_tokens = averages["generated_tokens"]
        if average_total_tokens is not None and average_generated_tokens is not None:
            pricing = MODEL_COST_PER_MILLION_TOKENS[model]
            average_cost = (
                (average_total_tokens - average_generated_tokens) * pricing["input"]
                + average_generated_tokens * pricing["output"]
            ) / 1_000_000
    return {
        "avg_total_tokens": format_tokens(averages["total_tokens"]),
        "avg_generated_tokens": format_tokens(averages["generated_tokens"]),
        "avg_cost": "N/A" if average_cost is None else f"${average_cost:.2f}",
    }


def benchmark_names(base_dir: Path) -> list[str]:
    """Return the benchmark names represented in result files under a run directory."""
    benchmarks: set[str] = set()
    for metrics_path in get_metrics_paths(base_dir):
        results_path = metrics_path.parent / "results.csv"
        results = pd.read_csv(results_path)
        if "benchmark" not in results.columns:
            raise ValueError(f"Missing benchmark column in {results_path}")
        if results["benchmark"].isna().any():
            raise ValueError(f"Missing benchmark value in {results_path}")
        benchmarks.update(results["benchmark"].astype(str))
    if not benchmarks:
        raise ValueError(f"No benchmark values found under {base_dir}")
    return sorted(benchmarks)


def agent_model_names(base_dir: Path) -> list[str]:
    """Return the task agent model names represented in result files under a base directory."""
    models: set[str] = set()
    for metrics_path in get_metrics_paths(base_dir):
        results_path = metrics_path.parent / "results.csv"
        results = pd.read_csv(results_path)
        if "model" not in results.columns:
            raise ValueError(f"Missing model column in {results_path}")
        if results["model"].isna().any():
            raise ValueError(f"Missing model value in {results_path}")
        models.update(results["model"].astype(str))
    if not models:
        raise ValueError(f"No model values found under {base_dir}")
    return sorted(models)


def normalized_difficulties(difficulties: pd.Series) -> pd.Series:
    """Combine difficulty values longer than one hour into a single group."""
    return difficulties.replace({"1-4 hours": "> 1 hour", ">4 hours": "> 1 hour"})


def difficulty_names(base_dir: Path) -> list[str | None]:
    """Return the normalized difficulty groups represented under a base directory."""
    difficulties: set[str | None] = set()
    for metrics_path in get_metrics_paths(base_dir):
        results_path = metrics_path.parent / "results.csv"
        results = pd.read_csv(results_path)
        if "difficulty" not in results.columns:
            raise ValueError(f"Missing difficulty column in {results_path}")
        normalized = normalized_difficulties(results["difficulty"])
        difficulties.update(None if pd.isna(value) else str(value) for value in normalized)
    if not difficulties:
        raise ValueError(f"No difficulty values found under {base_dir}")
    return sorted(difficulties, key=lambda value: (value is None, value or ""))


def build_results_table(
    base_dir: Path,
    *,
    shared_subset: bool = False,
    benchmark: str | None = None,
    model: str | None = None,
    difficulty: str | None = None,
    filter_by_difficulty: bool = False,
    min_n: int | None = None,
) -> pd.DataFrame:
    if min_n is not None and min_n < 1:
        raise ValueError(f"min_n must be at least 1, got {min_n}")

    metrics_paths = get_metrics_paths(base_dir)
    results_by_run_dir = {
        metrics_path.parent: pd.read_csv(metrics_path.parent / "results.csv") for metrics_path in metrics_paths
    }

    if min_n is not None:
        retained_results_by_run_dir: dict[Path, pd.DataFrame] = {}
        for run_dir, results in results_by_run_dir.items():
            if len(results) < min_n:
                logging.info("Excluding %s: %d results is below --min-n %d", run_dir, len(results), min_n)
                continue
            retained_results_by_run_dir[run_dir] = results
        results_by_run_dir = retained_results_by_run_dir
        if not results_by_run_dir:
            raise ValueError(f"No runs under {base_dir} have at least {min_n} results")

    if shared_subset:
        keys_by_run_dir: dict[Path, set[tuple[Any, ...]]] = {}
        for run_dir, results in results_by_run_dir.items():
            keys = set(results[SHARED_SUBSET_COLUMNS].itertuples(index=False, name=None))
            if len(keys) != len(results):
                raise ValueError(f"Duplicate instance/model pairs in {run_dir / 'results.csv'}")
            keys_by_run_dir[run_dir] = keys

        result_subsets: list[set[tuple[Any, ...]]] = []
        for run_dir, subset in keys_by_run_dir.items():
            if len(subset) == 0:
                logging.warning(f"No results for {run_dir}, not including in shared subset")
            else:
                result_subsets.append(subset)
        shared_keys = set.intersection(*result_subsets)
        if not shared_keys:
            raise ValueError(f"No shared instance/model pairs found across runs under {base_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir, results in results_by_run_dir.items():
        metrics_path = run_dir / "metrics.json"
        run_estimator_model = read_estimator_model_name(run_dir)
        if benchmark is not None:
            if "benchmark" not in results.columns:
                raise ValueError(f"Missing benchmark column in {run_dir / 'results.csv'}")
            results = results[results["benchmark"] == benchmark]
            if results.empty:
                raise ValueError(f"No results for benchmark={benchmark!r} in {run_dir / 'results.csv'}")
        if model is not None:
            if "model" not in results.columns:
                raise ValueError(f"Missing model column in {run_dir / 'results.csv'}")
            results = results[results["model"] == model]
            if results.empty:
                raise ValueError(f"No results for model={model!r} in {run_dir / 'results.csv'}")
        if filter_by_difficulty:
            if "difficulty" not in results.columns:
                raise ValueError(f"Missing difficulty column in {run_dir / 'results.csv'}")
            normalized = normalized_difficulties(results["difficulty"])
            matches_difficulty = normalized.isna() if difficulty is None else normalized == difficulty
            results = results[matches_difficulty]
            if results.empty:
                raise ValueError(f"No results for difficulty={difficulty!r} in {run_dir / 'results.csv'}")
        if shared_subset:
            result_keys = results[SHARED_SUBSET_COLUMNS].apply(tuple, axis=1)
            results = results[result_keys.isin(shared_keys)]

        if shared_subset:
            metrics = metrics_from_df(results).model_dump()
        elif benchmark is not None or model is not None or filter_by_difficulty:
            metrics = metrics_from_df(results).model_dump()
        else:
            metrics = read_metrics(metrics_path)

        rows.append(
            {
                "run": run_dir.relative_to(base_dir).as_posix(),
                "estimator_model": run_estimator_model,
                **metrics,
                **read_average_usage(results, run_dir / "results.csv", run_estimator_model),
            }
        )

    df = pd.DataFrame(rows).sort_values("run")
    ordered_columns = ["run", "estimator_model", *[column for column in CORE_METRIC_ORDER if column in df.columns]]
    ordered_columns.extend(
        column for column in df.columns if column not in ordered_columns and column not in USAGE_SUMMARY_ORDER
    )
    ordered_columns.extend(USAGE_SUMMARY_ORDER)
    return df[ordered_columns].rename(columns={**METRIC_COLUMN_LABELS, **USAGE_COLUMN_LABELS})


def simple_results_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return the compact result columns requested by the --simple CLI view."""
    missing_columns = set(SIMPLE_RESULT_COLUMNS) - set(results.columns)
    if missing_columns:
        raise ValueError(f"Missing columns required by --simple: {sorted(missing_columns)}")
    return results[SIMPLE_RESULT_COLUMNS]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.by_benchmark or args.by_model or args.by_difficulty:
        if args.csv:
            raise ValueError("--csv cannot be combined with a split flag")
        if args.output_path is not None:
            raise ValueError("--output_path cannot be combined with a split flag")
        if args.by_benchmark:
            split_label = "Benchmark"
            split_values: Sequence[str | None] = benchmark_names(args.base_dir)
        elif args.by_model:
            split_label = "Model"
            split_values = agent_model_names(args.base_dir)
        else:
            split_label = "Difficulty"
            split_values = difficulty_names(args.base_dir)
        for split_value in split_values:
            df = build_results_table(
                args.base_dir,
                shared_subset=args.shared_subset,
                benchmark=split_value if args.by_benchmark else None,
                model=split_value if args.by_model else None,
                difficulty=split_value if args.by_difficulty else None,
                filter_by_difficulty=args.by_difficulty,
                min_n=args.min_n,
            )
            if args.simple:
                df = simple_results_table(df)
            if args.tsv:
                print(f"{split_label}: {split_value}")
                print(df.to_csv(index=False, sep="\t", float_format=f"%{FLOAT_FORMAT}"), end="\n")
                continue
            print(f"{split_label}: {split_value}")
            print(df.to_markdown(index=False, floatfmt=FLOAT_FORMAT))
        return

    df = build_results_table(args.base_dir, shared_subset=args.shared_subset, min_n=args.min_n)
    if args.simple:
        df = simple_results_table(df)
    if args.csv:
        print(df.to_csv(index=False, float_format=f"%{FLOAT_FORMAT}"), end="")
        return
    if args.tsv:
        print(df.to_csv(index=False, sep="\t", float_format=f"%{FLOAT_FORMAT}"), end="")
        return

    output_path = args.output_path or args.base_dir / DEFAULT_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format=f"%{FLOAT_FORMAT}")
    print(f"Saved results to {output_path}")
    print(df.to_markdown(index=False, floatfmt=FLOAT_FORMAT))


def cli() -> None:
    main()
