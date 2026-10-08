import argparse
import json
import logging
import math
import traceback
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml
from datasets import Dataset, load_dataset
from tqdm import tqdm

from crg_ce.estimators.base_estimator import ConfEstimationOutput, output_path_for_item
from crg_ce.estimators.openhands.config import OHGSNEstimatorConfig, output_dir_for_run_config
from crg_ce.experiments.batch_run_estimate_confidence import ConfidenceEstimationRunConfig
from crg_ce.graph.data_types import ConfidenceGraph
from crg_ce.metrics import metrics_from_df

DEFAULT_WORKERS = 4
METRIC_ABS_TOLERANCE = 1e-5
PROVENANCE_FIELDS = (
    "problem_statement",
    "conversation_archive_path",
    "benchmark",
    "resolved",
    "trajectory_type",
)
OUTPUT_RESULT_FIELDS = {
    "confidence": "estimated_confidence",
    "total_tokens": "total_tokens",
    "generated_tokens": "generated_tokens",
    "cost": "cost",
}


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ItemAudit:
    instance_id: str
    model: str
    output_dir: str
    status: Literal["ok", "warning", "error"]
    issues: tuple[Issue, ...] = ()
    traceback: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a completed confidence-estimation run before reporting its metrics."
    )
    parser.add_argument("run_config", type=Path, help="Target run YAML used to infer the output directory.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset path or repository ID.")
    parser.add_argument("--split", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def _issue(code: str, message: str, *, warning: bool = False, **details: Any) -> Issue:
    return Issue("warning" if warning else "error", code, message, details)


def _config_differences(expected: Any, actual: Any, path: str = "") -> list[dict[str, Any]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[dict[str, Any]] = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in expected:
                differences.append({"field": child_path, "expected": "<missing>", "actual": actual[key]})
            elif key not in actual:
                differences.append({"field": child_path, "expected": expected[key], "actual": "<missing>"})
            else:
                differences.extend(_config_differences(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            child_path = f"{path}[{index}]"
            if index >= len(expected):
                differences.append({"field": child_path, "expected": "<missing>", "actual": actual[index]})
            elif index >= len(actual):
                differences.append({"field": child_path, "expected": expected[index], "actual": "<missing>"})
            else:
                differences.extend(_config_differences(expected[index], actual[index], child_path))
        return differences
    if expected != actual or type(expected) is not type(actual):
        return [{"field": path or "<root>", "expected": expected, "actual": actual}]
    return []


def _load_target_config(config_path: Path) -> tuple[str, dict[str, Any], ConfidenceEstimationRunConfig]:
    config_text = config_path.read_text()
    config_data = yaml.safe_load(config_text)
    if not isinstance(config_data, dict):
        raise TypeError(f"Expected a YAML object in {config_path}, got {type(config_data).__name__}")
    return config_text, config_data, ConfidenceEstimationRunConfig.model_validate(config_data)


def _audit_saved_config(
    *,
    target_text: str,
    target_data: dict[str, Any],
    saved_config_path: Path,
) -> list[Issue]:
    # Makes sure the run_config.yaml we pass and the run_config.yaml saved in outputs/ are identical,
    # reporting any differences
    if not saved_config_path.is_file():
        return [_issue("missing_saved_config", f"Missing saved run config: {saved_config_path}")]
    saved_text = saved_config_path.read_text()
    saved_data = yaml.safe_load(saved_text)
    if not isinstance(saved_data, dict):
        return [_issue("invalid_saved_config", f"Expected a YAML object in {saved_config_path}")]

    differences = _config_differences(target_data, saved_data)
    if differences:
        return [
            _issue(
                "saved_config_field_mismatch",
                "Saved run configuration differs from the passed run configuration",
                differences=differences,
            )
        ]
    if saved_text != target_text:
        return [
            _issue(
                "saved_config_text_mismatch",
                "Saved and passed run configurations are semantically equal but not text-identical",
                warning=True,
            )
        ]
    return []


def _instance_id(row: Mapping[str, Any]) -> tuple[str, str]:
    instance_id = row["instance_id"]
    model = row["model"]
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(f"instance_id must be a nonblank string, got {instance_id!r}")
    if not isinstance(model, str) or not model:
        raise ValueError(f"model must be a nonblank string, got {model!r}")
    return instance_id, model


def _is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, bool) else False


def _values_equal(expected: Any, actual: Any) -> bool:
    if expected is None:
        return _is_missing(actual)
    if isinstance(expected, list | dict):
        return bool(actual == expected or actual == str(expected) or actual == json.dumps(expected))
    if isinstance(expected, float) and math.isnan(expected):
        return _is_missing(actual)
    return bool(expected == actual)


def _numbers_equal(expected: Any, actual: Any) -> bool:
    try:
        expected_number = float(expected)
        actual_number = float(actual)
    except (TypeError, ValueError):
        return False
    if math.isnan(expected_number) or math.isnan(actual_number):
        return math.isnan(expected_number) and math.isnan(actual_number)
    return math.isclose(expected_number, actual_number, rel_tol=1e-12, abs_tol=1e-12)


def _metrics_equal(expected: Any, actual: Any) -> bool:
    try:
        expected_number = float(expected)
        actual_number = float(actual)
    except (TypeError, ValueError):
        return False
    if math.isnan(expected_number) or math.isnan(actual_number):
        return math.isnan(expected_number) and math.isnan(actual_number)
    return math.isclose(expected_number, actual_number, rel_tol=1e-12, abs_tol=METRIC_ABS_TOLERANCE)


def _audit_graph(graph_path: Path, output: ConfEstimationOutput) -> list[Issue]:
    # Makes sure each oh_gsn output has a graph whose root confidence matches output.json
    if not graph_path.is_file():
        return [_issue("missing_graph", f"Missing graph artifact: {graph_path}")]
    graph = ConfidenceGraph.model_validate_json(graph_path.read_text())
    if graph.goal_zero_node_id is None:
        return [_issue("missing_graph_root", f"Graph has no goal_zero_node_id: {graph_path}")]
    root = next(node for node in graph.nodes if node.id == graph.goal_zero_node_id)
    if not _numbers_equal(root.confidence, output.confidence):
        return [
            _issue(
                "graph_output_confidence_mismatch",
                "Graph root confidence does not match output.json",
                graph_path=str(graph_path),
                graph_confidence=root.confidence,
                output_confidence=output.confidence,
            )
        ]
    return []


def audit_item(
    *,
    dataset_row: Mapping[str, Any],
    result_row: Mapping[str, Any] | None,
    failure_record: Mapping[str, Any] | None = None,
    run_dir: Path,
    check_graph: bool,
) -> ItemAudit:
    # Makes sure each dataset item has mutually consistent dataset, results.csv, output.json,
    # and (for oh_gsn runs) graph.json records
    instance_id, model = _instance_id(dataset_row)
    item_output_dir = run_dir / instance_id / model
    issues: list[Issue] = []
    try:
        if result_row is None:
            failure_details: dict[str, Any] = {}
            if failure_record is not None:
                failure_error = failure_record["error"]
                if not isinstance(failure_error, str):
                    raise TypeError(f"failure error must be a string, got {type(failure_error).__name__}")
                failure_details["failure"] = {
                    "error": failure_error[:120],
                    "error_truncated": len(failure_error) > 120,
                }
            issues.append(
                _issue(
                    "missing_prediction",
                    "Dataset item is absent from results.csv",
                    **failure_details,
                )
            )

        output_path = output_path_for_item(run_dir, instance_id, model)
        output: ConfEstimationOutput | None = None
        if not output_path.is_file():
            issues.append(_issue("missing_output", f"Missing estimator output: {output_path}"))
        else:
            output = ConfEstimationOutput.model_validate_json(output_path.read_text())

        # make sure the dataset and our results agree on key data fields:
        # - problem_statement, conversation_archive_path, benchmark, resolved, [trajectory_type]
        if result_row is not None:
            for provenance_field in PROVENANCE_FIELDS:
                if provenance_field not in dataset_row:
                    continue
                if provenance_field not in result_row:
                    issues.append(
                        _issue(
                            "missing_result_provenance_field",
                            f"results.csv lacks dataset field {provenance_field!r}",
                        )
                    )
                    continue
                if not _values_equal(dataset_row[provenance_field], result_row[provenance_field]):
                    issues.append(
                        _issue(
                            "result_dataset_mismatch",
                            f"results.csv {provenance_field} differs from the current dataset",
                            field=provenance_field,
                            dataset_value=dataset_row[provenance_field],
                            result_value=result_row[provenance_field],
                        )
                    )

        if result_row is not None and output is not None:
            for output_field, result_field in OUTPUT_RESULT_FIELDS.items():
                if result_field not in result_row:
                    issues.append(_issue("missing_result_output_field", f"results.csv lacks {result_field!r}"))
                    continue
                output_value = getattr(output, output_field)
                if not _numbers_equal(output_value, result_row[result_field]):
                    issues.append(
                        _issue(
                            "result_output_mismatch",
                            f"results.csv {result_field} differs from output.json",
                            field=result_field,
                            output_value=output_value,
                            result_value=result_row[result_field],
                        )
                    )

        if check_graph and output is not None:
            issues.extend(_audit_graph(item_output_dir / "graph.json", output))
    except Exception as error:
        issues.append(_issue("item_audit_exception", f"{type(error).__name__}: {error}"))
        return ItemAudit(
            instance_id,
            model,
            str(item_output_dir),
            "error",
            tuple(issues),
            "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )

    status: Literal["ok", "warning", "error"] = "ok"
    if any(issue.severity == "error" for issue in issues):
        status = "error"
    elif issues:
        status = "warning"
    return ItemAudit(instance_id, model, str(item_output_dir), status, tuple(issues))


def _audit_items(
    rows: Sequence[Mapping[str, Any]],
    *,
    results_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    failures_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
    run_dir: Path,
    check_graph: bool,
    workers: int,
) -> list[ItemAudit]:
    def audit(row: Mapping[str, Any]) -> ItemAudit:
        return audit_item(
            dataset_row=row,
            result_row=results_by_key.get(_instance_id(row)),
            failure_record=failures_by_key.get(_instance_id(row)),
            run_dir=run_dir,
            check_graph=check_graph,
        )

    if workers == 1:
        return [audit(row) for row in tqdm(rows, desc="audit run")]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(executor.map(audit, rows), total=len(rows), desc="audit run"))


def _duplicate_instance_ids(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    counts = Counter(_instance_id(row) for row in rows)
    return [key for key, count in counts.items() if count > 1]


def _audit_metrics(results: pd.DataFrame, metrics_path: Path) -> list[Issue]:
    # Makes sure metrics.json contains the metrics parse_results.py would recompute from results.csv
    if not metrics_path.is_file():
        return [_issue("missing_metrics", f"Missing metrics file used by parse_results.py: {metrics_path}")]
    try:
        saved_metrics = json.loads(metrics_path.read_text())
        recomputed_metrics = metrics_from_df(results).model_dump()
    except Exception as error:
        return [_issue("metrics_validation_failed", f"{type(error).__name__}: {error}")]
    mismatches = [
        {
            "field": field,
            "saved": saved_metrics.get(field),
            "recomputed": recomputed_value,
        }
        for field, recomputed_value in recomputed_metrics.items()
        if field not in saved_metrics or not _metrics_equal(saved_metrics[field], recomputed_value)
    ]
    if mismatches:
        return [
            _issue(
                "stale_metrics",
                "metrics.json differs from metrics recomputed from results.csv",
                mismatches=mismatches,
            )
        ]
    return []


def _read_failures(run_dir: Path) -> list[dict[str, Any]]:
    failures_path = run_dir / "failures.json"
    if not failures_path.is_file():
        return []
    failures = json.loads(failures_path.read_text())
    if not isinstance(failures, list):
        raise TypeError(f"Expected a list in {failures_path}, got {type(failures).__name__}")
    if not all(isinstance(failure, dict) for failure in failures):
        raise TypeError(f"Expected every entry in {failures_path} to be an object")
    return failures


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    # Makes sure the run_config.yaml we pass and the run_config.yaml saved in outputs/ are identical,
    # reporting any differences
    target_text, target_data, config = _load_target_config(args.run_config)
    run_dir = config.output.output_dir or output_dir_for_run_config(args.run_config)
    report_path = args.report_path or run_dir / "audit_run.json"
    global_issues = _audit_saved_config(
        target_text=target_text,
        target_data=target_data,
        saved_config_path=run_dir / "run_config.yaml",
    )

    # Makes sure the explicitly requested dataset and split match the passed run configuration
    if config.dataset.path != args.dataset:
        global_issues.append(
            _issue(
                "dataset_argument_config_mismatch",
                "--dataset differs from dataset.path in the passed run config",
                argument=args.dataset,
                config=config.dataset.path,
            )
        )
    if config.dataset.split != args.split:
        global_issues.append(
            _issue(
                "split_argument_config_mismatch",
                "--split differs from dataset.split in the passed run config",
                argument=args.split,
                config=config.dataset.split,
            )
        )

    loaded_dataset = load_dataset(
        args.dataset,
        name=config.dataset.name,
        split=args.split,
        revision=config.dataset.revision,
    )
    if not isinstance(loaded_dataset, Dataset):
        raise TypeError(f"Expected a materialized Dataset, got {type(loaded_dataset).__name__}")
    dataset_rows = loaded_dataset.to_list()

    # Makes sure every dataset item has a unique `instance_id` for matching it to one result and output
    duplicate_dataset_keys = _duplicate_instance_ids(dataset_rows)
    if duplicate_dataset_keys:
        global_issues.append(
            _issue(
                "duplicate_dataset_keys",
                "Dataset contains duplicate (instance_id, model) keys",
                count=len(duplicate_dataset_keys),
                examples=duplicate_dataset_keys[:20],
            )
        )

    # make sure we have a recorded results.csv
    results_path = run_dir / "results.csv"
    if results_path.is_file():
        results = pd.read_csv(results_path)
        result_rows = cast(list[dict[str, Any]], results.to_dict(orient="records"))
    else:
        results = pd.DataFrame()
        result_rows = []
        global_issues.append(_issue("missing_results", f"Missing parse-results source: {results_path}"))

    # Makes sure results.csv has unique keys and contains no results from another dataset or split
    duplicate_result_keys = _duplicate_instance_ids(result_rows)
    if duplicate_result_keys:
        global_issues.append(
            _issue(
                "duplicate_result_keys",
                "results.csv contains duplicate (instance_id, model) keys",
                count=len(duplicate_result_keys),
                examples=duplicate_result_keys[:20],
            )
        )
    results_by_key = {_instance_id(row): row for row in result_rows}
    dataset_keys = {_instance_id(row) for row in dataset_rows}
    extra_result_keys = sorted(set(results_by_key) - dataset_keys)
    # this makes sure we do not have orphaned rows CONTRIBUTING TO OUR SCORES. This is a serious error if so, even
    # though having orphaned results is expected in some cases (they must not contribute to reported scores)
    if extra_result_keys:
        global_issues.append(
            _issue(
                "extra_results",
                "results.csv contains predictions not present in the requested dataset split",
                count=len(extra_result_keys),
                examples=extra_result_keys[:20],
            )
        )

    if not results.empty:
        # make sure the computed, final result metrics match what occurs on re-compute
        global_issues.extend(_audit_metrics(results, run_dir / "metrics.json"))

    estimator = config.estimator
    check_graph: bool = isinstance(estimator, OHGSNEstimatorConfig)
    # Makes sure an oh_gsn run uses the agentic graph generator required by this audit (others are outdated)
    if check_graph and (estimator.graph_generator is None or estimator.graph_generator.generator_type != "agentic"):
        global_issues.append(
            _issue(
                "non_agentic_oh_gsn_generator",
                "oh_gsn audit requires estimator.graph_generator.generator_type == 'agentic'",
            )
        )

    failures = _read_failures(run_dir)
    duplicate_failure_keys = _duplicate_instance_ids(failures)
    if duplicate_failure_keys:
        global_issues.append(
            _issue(
                "duplicate_failure_keys",
                "failures.json contains duplicate (instance_id, model) keys",
                count=len(duplicate_failure_keys),
                examples=duplicate_failure_keys[:20],
            )
        )
    failures_by_key = {_instance_id(failure): failure for failure in failures}

    # line-by-line: audit each row
    item_audits = _audit_items(
        dataset_rows,
        results_by_key=results_by_key,
        failures_by_key=failures_by_key,
        run_dir=run_dir,
        check_graph=check_graph,
        workers=args.workers,
    )
    missing_prediction_count = sum(
        any(issue.code == "missing_prediction" for issue in audit.issues) for audit in item_audits
    )

    # Makes sure the run directory has no output.json artifacts belonging to other dataset items
    expected_output_paths = {output_path_for_item(run_dir, *_instance_id(row)).resolve() for row in dataset_rows}
    actual_output_paths = {path.resolve() for path in run_dir.rglob("output.json")}
    orphan_output_paths = sorted(str(path) for path in actual_output_paths - expected_output_paths)
    if orphan_output_paths:
        global_issues.append(
            _issue(
                "orphan_outputs",
                "Run directory contains output.json files outside the requested dataset split",
                warning=True,
                count=len(orphan_output_paths),
                examples=orphan_output_paths[:20],
            )
        )

    # Makes sure failures recorded by the production runner are visible in the audit report
    if failures:
        global_issues.append(
            _issue(
                "recorded_failures",
                "Run has entries in failures.json",
                warning=missing_prediction_count == 0,
                count=len(failures),
            )
        )

    all_issues = [*global_issues, *(issue for audit in item_audits for issue in audit.issues)]
    report = {
        "run_config": str(args.run_config),
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "split": args.split,
        "summary": {
            "dataset_items": len(dataset_rows),
            "result_rows": len(result_rows),
            "missing_predictions": missing_prediction_count,
            "item_status_counts": dict(Counter(audit.status for audit in item_audits)),
            "global_error_count": sum(issue.severity == "error" for issue in global_issues),
            "global_warning_count": sum(issue.severity == "warning" for issue in global_issues),
        },
        "global_issues": [asdict(issue) for issue in global_issues],
        "items": [asdict(audit) for audit in item_audits if audit.status != "ok"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Audit summary: %s; report: %s", report["summary"], report_path)

    has_errors = any(issue.severity == "error" for issue in all_issues)
    has_warnings = any(issue.severity == "warning" for issue in all_issues)
    if has_errors or (args.fail_on_warning and has_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
