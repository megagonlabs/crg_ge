import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from crg_ce.experiments.parse_results import (
    DEFAULT_BASE_DIR,
    FLOAT_FORMAT,
    SHARED_SUBSET_COLUMNS,
    agent_model_names,
    benchmark_names,
    difficulty_names,
    get_metrics_paths,
    normalized_difficulties,
    read_estimator_model_name,
)
from crg_ce.metrics import adaptive_expected_cal_error, expected_cal_error

DEFAULT_BINS = [5, 10, 15, 20]
DEFAULT_OUTPUT_FILENAME = "ece_sweep.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=DEFAULT_BINS,
        help=f"Bin counts to evaluate (default: {' '.join(map(str, DEFAULT_BINS))})",
    )
    output_format = parser.add_mutually_exclusive_group()
    output_format.add_argument(
        "--csv", action="store_true", help="Write CSV to stdout instead of saving a sweep file"
    )
    output_format.add_argument(
        "--tsv", action="store_true", help="Write TSV to stdout instead of saving a sweep file"
    )
    parser.add_argument(
        "--shared-subset",
        action="store_true",
        help="Compute results for the instance/model pairs present in every run",
    )
    split_by = parser.add_mutually_exclusive_group()
    split_by.add_argument(
        "--by-benchmark",
        action="store_true",
        help="Print a separate ECE sweep table for each benchmark",
    )
    split_by.add_argument(
        "--by-model",
        action="store_true",
        help="Print a separate ECE sweep table for each agent model",
    )
    split_by.add_argument(
        "--by-difficulty",
        action="store_true",
        help="Print a separate ECE sweep table for each normalized difficulty",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=None,
        help="Exclude runs with fewer than N total rows before filtering or taking a shared subset",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Omit the estimator model column",
    )
    return parser.parse_args(argv)


def validate_bins(bins: Sequence[int]) -> list[int]:
    if any(n_bins < 1 for n_bins in bins):
        raise ValueError(f"Bin counts must be positive, got {list(bins)}")
    if len(set(bins)) != len(bins):
        raise ValueError(f"Bin counts must be unique, got {list(bins)}")
    return list(bins)


def build_ece_sweep_table(
    base_dir: Path,
    bins: Sequence[int] = DEFAULT_BINS,
    *,
    shared_subset: bool = False,
    benchmark: str | None = None,
    model: str | None = None,
    difficulty: str | None = None,
    filter_by_difficulty: bool = False,
    min_n: int | None = None,
) -> pd.DataFrame:
    bins = validate_bins(bins)
    if min_n is not None and min_n < 1:
        raise ValueError(f"min_n must be at least 1, got {min_n}")

    metrics_paths = get_metrics_paths(base_dir)
    results_by_run_dir = {
        metrics_path.parent: pd.read_csv(metrics_path.parent / "results.csv") for metrics_path in metrics_paths
    }

    if min_n is not None:
        results_by_run_dir = {
            run_dir: results
            for run_dir, results in results_by_run_dir.items()
            if _retain_run(run_dir, results, min_n)
        }
        if not results_by_run_dir:
            raise ValueError(f"No runs under {base_dir} have at least {min_n} results")

    if shared_subset:
        keys_by_run_dir: dict[Path, set[tuple[Any, ...]]] = {}
        for run_dir, results in results_by_run_dir.items():
            keys = set(results[SHARED_SUBSET_COLUMNS].itertuples(index=False, name=None))
            if len(keys) != len(results):
                raise ValueError(f"Duplicate instance/model pairs in {run_dir / 'results.csv'}")
            keys_by_run_dir[run_dir] = keys

        nonempty_key_sets = [keys for keys in keys_by_run_dir.values() if keys]
        if not nonempty_key_sets:
            raise ValueError(f"No result rows found under {base_dir}")
        shared_keys = set.intersection(*nonempty_key_sets)
        if not shared_keys:
            raise ValueError(f"No shared instance/model pairs found across runs under {base_dir}")

    rows: list[dict[str, Any]] = []
    for run_dir, results in results_by_run_dir.items():
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

        if results[SHARED_SUBSET_COLUMNS].duplicated().any():
            raise ValueError(f"Duplicate instance/model pairs in {run_dir / 'results.csv'}")
        confidence = results["estimated_confidence"].to_numpy()
        correctness = results["resolved"].to_numpy()
        row: dict[str, Any] = {
            "run": run_dir.relative_to(base_dir).as_posix(),
            "estimator_model": read_estimator_model_name(run_dir),
            "n": len(results),
        }
        row.update(
            {
                f"ece_{n_bins} (↓)": expected_cal_error(
                    confidence=confidence,
                    correctness=correctness,
                    n_bins=n_bins,
                )
                for n_bins in bins
            }
        )
        row.update(
            {
                f"adaptive_ece_{n_bins} (↓)": adaptive_expected_cal_error(
                    confidence=confidence,
                    correctness=correctness,
                    n_bins=n_bins,
                )
                for n_bins in bins
            }
        )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("run")


def _retain_run(run_dir: Path, results: pd.DataFrame, min_n: int) -> bool:
    if len(results) >= min_n:
        return True
    logging.info("Excluding %s: %d results is below --min-n %d", run_dir, len(results), min_n)
    return False


def _print_table(df: pd.DataFrame, *, tsv: bool) -> None:
    if tsv:
        print(df.to_csv(index=False, sep="\t", float_format=f"%{FLOAT_FORMAT}"), end="")
    else:
        print(df.to_markdown(index=False, floatfmt=FLOAT_FORMAT))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    bins = validate_bins(args.bins)
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
            df = build_ece_sweep_table(
                args.base_dir,
                bins,
                shared_subset=args.shared_subset,
                benchmark=split_value if args.by_benchmark else None,
                model=split_value if args.by_model else None,
                difficulty=split_value if args.by_difficulty else None,
                filter_by_difficulty=args.by_difficulty,
                min_n=args.min_n,
            )
            if args.simple:
                df = df.drop(columns="estimator_model")
            print(f"{split_label}: {split_value}")
            _print_table(df, tsv=args.tsv)
            if args.tsv:
                print()
        return

    df = build_ece_sweep_table(
        args.base_dir,
        bins,
        shared_subset=args.shared_subset,
        min_n=args.min_n,
    )
    if args.simple:
        df = df.drop(columns="estimator_model")
    if args.csv:
        print(df.to_csv(index=False, float_format=f"%{FLOAT_FORMAT}"), end="")
        return
    if args.tsv:
        _print_table(df, tsv=True)
        return

    output_path = args.output_path or args.base_dir / DEFAULT_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format=f"%{FLOAT_FORMAT}")
    print(f"Saved results to {output_path}")
    _print_table(df, tsv=False)


def cli() -> None:
    main()
