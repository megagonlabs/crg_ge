import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset

from crg_ce.estimators.base_estimator import ConfEstimationOutput
from crg_ce.estimators.openhands.oh_gsn_estimator import _get_validated_goal_zero
from crg_ce.experiments.batch_run_estimate_confidence import ConfidenceEstimationRunConfig
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.metrics import ConfMetrics, metrics_from_df

DEFAULT_RESULTS_FILENAME = "results.csv"
DEFAULT_METRICS_FILENAME = "metrics.json"
RUN_CONFIG_FILENAME = "run_config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate results and metrics for one run or every run under a directory."
    )
    parser.add_argument("path", type=Path, help="Run directory or directory containing runs")
    parser.add_argument("--results-filename", default=DEFAULT_RESULTS_FILENAME)
    parser.add_argument("--metrics-filename", default=DEFAULT_METRICS_FILENAME)
    return parser.parse_args()


def read_run_config(run_dir: Path) -> ConfidenceEstimationRunConfig:
    config_path = run_dir / RUN_CONFIG_FILENAME
    config_data = yaml.safe_load(config_path.read_text())
    if not isinstance(config_data, dict):
        raise TypeError(f"Expected config object in {config_path}, got {type(config_data).__name__}")
    return ConfidenceEstimationRunConfig.model_validate(config_data)


def get_instance_id_and_model_from_path(run_dir: Path, graph_or_output_path: Path) -> tuple[str, str]:
    relative_parent = graph_or_output_path.parent.relative_to(run_dir)
    if len(relative_parent.parts) < 2:
        raise ValueError(
            f"Expected graph path to include instance_id and model under {run_dir}: {graph_or_output_path}"
        )
    instance_id = relative_parent.parts[0]
    model = Path(*relative_parent.parts[1:]).as_posix()
    return instance_id, model


def read_graph_confidence(graph_path: Path) -> float:
    graph = ConfidenceGraph.model_validate_json(graph_path.read_text())
    goal_zero = _get_validated_goal_zero(graph)
    if goal_zero.confidence < 0:
        raise ValueError(f"Goal-zero confidence was not populated in {graph_path}: node_id={goal_zero.id}")
    return goal_zero.confidence


def read_conf_est_output(output_path: Path) -> dict[str, Any]:
    result = ConfEstimationOutput.model_validate_json(output_path.read_text())
    if result.confidence < 0:
        raise ValueError("-1 confidence was saved")
    result_model = result.model_dump()
    result_model["estimated_confidence"] = result.confidence
    return result_model


def load_dataset_rows_by_key(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    cfg = read_run_config(run_dir)
    dataset_cfg = cfg.dataset.model_copy(update={"slice": None, "sub_sample_n": 0})
    dataset: Dataset = dataset_cfg.load_dataset()
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for item in dataset.to_list():
        if not isinstance(item, dict):
            raise TypeError(f"Expected dataset row object, got {type(item).__name__}")
        key = (item["instance_id"], item["model"])
        if key in rows_by_key:
            raise ValueError(f"Duplicate dataset row for instance/model: {key}")
        rows_by_key[key] = item
    return rows_by_key


def build_partial_results_df(run_dir: Path) -> pd.DataFrame:
    output_paths: list[Path] = sorted(run_dir.rglob("output.json"))
    graph_paths = sorted(run_dir.rglob("graph.json"))

    rows_by_key = load_dataset_rows_by_key(run_dir)
    result_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for output_path in output_paths:
        key = get_instance_id_and_model_from_path(run_dir, output_path)
        seen_keys.add(key)
        dataset_row = rows_by_key.get(key)
        if not dataset_row:
            logging.warning("%s not in dataset", key)
            continue
        result_rows.append({**dataset_row, **read_conf_est_output(output_path)})

    for graph_path in graph_paths:
        key = get_instance_id_and_model_from_path(run_dir, graph_path)
        if key in seen_keys:
            continue

        dataset_row = rows_by_key.get(key)
        if not dataset_row:
            logging.warning("%s not in dataset", key)
            continue
        result_rows.append({**dataset_row, "estimated_confidence": read_graph_confidence(graph_path)})

    return pd.DataFrame(result_rows)


def generate_metrics(run_dir: Path, results_filename: str, metrics_filename: str) -> ConfMetrics:
    df = build_partial_results_df(run_dir)
    results_path = run_dir / results_filename
    df.to_csv(results_path)
    metrics = metrics_from_df(df)
    metrics_path = run_dir / metrics_filename
    metrics_path.write_text(metrics.model_dump_json(indent=4))
    return metrics


def find_run_dirs(path: Path) -> list[Path]:
    if (path / RUN_CONFIG_FILENAME).is_file():
        return [path]

    run_dirs = sorted(config_path.parent for config_path in path.rglob(RUN_CONFIG_FILENAME))
    if not run_dirs:
        raise ValueError(f"No {RUN_CONFIG_FILENAME} files found under {path}")
    return run_dirs


def cli() -> None:
    args = parse_args()
    run_dirs = find_run_dirs(args.path)
    for run_dir in run_dirs:
        metrics = generate_metrics(
            run_dir=run_dir,
            results_filename=args.results_filename,
            metrics_filename=args.metrics_filename,
        )
        metrics_path = run_dir / args.metrics_filename
        print(f"Saved metrics to {metrics_path}")
        print(metrics.get_metric_log())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli()
