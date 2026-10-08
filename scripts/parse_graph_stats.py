import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from crg_ce.utils.graphs import (
    GRAPH_LAYERS,
    GraphData,
    average_confidence,
    get_node_ids_by_layer,
    max_node_depth,
    read_graph,
)

DEFAULT_BASE_DIR = Path("outputs/graphs/runs/exploration/gsn_graph_gen")
DEFAULT_OUTPUT_FILENAME = "graph_stats_summary.csv"
FLOAT_FORMAT = ".2f"
CORE_STAT_COLUMNS = [
    "avg_nodes",
    "avg_num_goal",
    "avg_num_evid",
    "avg_depth",
    "max_depth",
    "p95_depth",
    "avg_goal_leaves",
    "avg_L0_nodes",
    "avg_L1_nodes",
    "avg_L2_nodes",
    "avg_L3+_nodes",
    "avg_L0_conf",
    "avg_L1_conf",
    "avg_L2_conf",
    "avg_L3+_conf",
    "avg_edges",
    "avg_goal_leaf_conf",
]
MISSING_VALUE = "-"
LAYERS = GRAPH_LAYERS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--csv", action="store_true", help="Write CSV to stdout instead of saving a summary file")
    parser.add_argument(
        "--by-benchmark", action="store_true", help="Print a separate graph stats table for each benchmark"
    )
    parser.add_argument(
        "--simple", action="store_true", help="Keep only aggregate node, depth, edge, and confidence columns"
    )
    return parser.parse_args()


def get_run_dirs(base_dir: Path) -> list[Path]:
    metrics_paths = sorted(base_dir.rglob("metrics.json"))
    if metrics_paths:
        return [metrics_path.parent for metrics_path in metrics_paths]

    graph_paths = sorted(base_dir.rglob("graph.json"))
    if not graph_paths:
        raise FileNotFoundError(f"No metrics.json or graph.json files found under {base_dir}")

    run_dirs: set[Path] = set()
    for graph_path in graph_paths:
        graph_parent = graph_path.parent
        relative_parent = graph_parent.relative_to(base_dir)
        if not relative_parent.parts:
            run_dirs.add(base_dir)
        else:
            run_dirs.add(base_dir / relative_parent.parts[0])
    return sorted(run_dirs)


def benchmark_names(base_dir: Path) -> list[str]:
    benchmarks: set[str] = set()
    for run_dir in get_run_dirs(base_dir):
        results_path = run_dir / "results.csv"
        results = pd.read_csv(results_path)
        if "benchmark" not in results.columns:
            raise ValueError(f"Missing benchmark column in {results_path}")
        if results["benchmark"].isna().any():
            raise ValueError(f"Missing benchmark value in {results_path}")
        benchmarks.update(results["benchmark"].astype(str))
    if not benchmarks:
        raise ValueError(f"No benchmark values found under {base_dir}")
    return sorted(benchmarks)


def build_graph_stats(graph: GraphData) -> dict[str, float]:
    relationship_counts = Counter(str(edge["relationship_type"]) for edge in graph["edges"])
    nodes_by_id = {node["id"]: node for node in graph["nodes"]}
    goal_node_ids = {node["id"] for node in graph["nodes"] if node["kind"] == "GSNGoalNode"}
    goal_only_graph = {
        "nodes": [node for node in graph["nodes"] if node["id"] in goal_node_ids],
        "edges": [
            edge for edge in graph["edges"] if edge["source"] in goal_node_ids and edge["target"] in goal_node_ids
        ],
        "goal_zero_node_id": graph.get("goal_zero_node_id"),
    }
    goal_parent_ids = {edge["target"] for edge in graph["edges"] if edge["source"] in goal_node_ids}
    goal_leaf_ids = goal_node_ids - goal_parent_ids
    node_ids_by_layer = get_node_ids_by_layer(graph)

    stats = {
        "nodes": len(graph["nodes"]),
        "num_goal": len(goal_node_ids),
        "num_evid": len(graph["nodes"]) - len(goal_node_ids),
        "depth": max_node_depth(goal_only_graph),
        "goal_leaves": len(goal_leaf_ids),
        "edges": len(graph["edges"]),
        "goal_leaf_conf": average_confidence([nodes_by_id[node_id] for node_id in goal_leaf_ids]),
    }
    stats.update({f"L{layer}_nodes": len(node_ids_by_layer[layer]) for layer in LAYERS})
    stats.update(
        {
            f"L{layer}_conf": average_confidence([nodes_by_id[node_id] for node_id in node_ids_by_layer[layer]])
            for layer in LAYERS
        }
    )
    stats.update(
        {
            f"relationship_type_{relationship_type}": count
            for relationship_type, count in sorted(relationship_counts.items())
        }
    )
    return stats


def summarize_graph_stats(graph_stats: list[dict[str, float]]) -> dict[str, float]:
    stat_names = sorted({stat_name for stats in graph_stats for stat_name in stats})
    summary: dict[str, float] = {}
    for stat_name in stat_names:
        values = [stats.get(stat_name, 0.0) for stats in graph_stats]
        finite_values = [value for value in values if not math.isnan(value)]
        summary[f"avg_{stat_name}"] = sum(finite_values) / len(finite_values) if finite_values else math.nan

    depths = [stats["depth"] for stats in graph_stats if not math.isnan(stats["depth"])]
    summary["max_depth"] = max(depths) if depths else math.nan
    summary["p95_depth"] = pd.Series(depths).quantile(0.95, interpolation="nearest") if depths else math.nan
    return summary


def build_missing_stats_row() -> dict[str, float]:
    return {column: math.nan for column in CORE_STAT_COLUMNS}


def build_results_table(base_dir: Path, *, simple: bool = False, benchmark: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in get_run_dirs(base_dir):
        graph_paths = sorted(run_dir.rglob("graph.json"))
        if benchmark is not None:
            results_path = run_dir / "results.csv"
            results = pd.read_csv(results_path)
            if "benchmark" not in results.columns:
                raise ValueError(f"Missing benchmark column in {results_path}")
            instance_ids = set(results.loc[results["benchmark"] == benchmark, "instance_id"])
            if not instance_ids:
                raise ValueError(f"No results for benchmark={benchmark!r} in {results_path}")
            graph_paths = [
                graph_path for graph_path in graph_paths if graph_path.relative_to(run_dir).parts[0] in instance_ids
            ]
        if not graph_paths:
            rows.append(
                {
                    "run": run_dir.relative_to(base_dir).as_posix(),
                    **build_missing_stats_row(),
                }
            )
            continue

        rows.append(
            {
                "run": run_dir.relative_to(base_dir).as_posix(),
                **summarize_graph_stats([build_graph_stats(read_graph(graph_path)) for graph_path in graph_paths]),
            }
        )

    df = pd.DataFrame(rows).sort_values("run")
    ordered_columns = ["run", *[column for column in CORE_STAT_COLUMNS if column in df.columns]]
    ordered_columns.extend(column for column in df.columns if column not in ordered_columns)
    df = df[ordered_columns]

    stat_columns = [column for column in df.columns if column != "run"]
    count_columns = [column for column in stat_columns if not column.endswith("_conf")]
    missing_rows = df[CORE_STAT_COLUMNS].isna().all(axis=1) if set(CORE_STAT_COLUMNS) <= set(df.columns) else []
    rows_with_graphs = ~missing_rows if len(missing_rows) else slice(None)
    df.loc[rows_with_graphs, count_columns] = df.loc[rows_with_graphs, count_columns].fillna(0.0)
    if simple:
        simple_columns = [
            "run",
            "avg_nodes",
            "avg_num_goal",
            "avg_num_evid",
            "avg_depth",
            "max_depth",
            "p95_depth",
            "avg_goal_leaves",
            "avg_edges",
        ]
        simple_columns.extend(f"avg_L{layer}_conf" for layer in LAYERS)
        simple_columns.append("avg_goal_leaf_conf")
        return df[[column for column in simple_columns if column in df.columns]]
    return df


def main() -> None:
    args = parse_args()
    if args.by_benchmark:
        if args.csv:
            raise ValueError("--csv cannot be combined with --by-benchmark")
        if args.output_path is not None:
            raise ValueError("--output_path cannot be combined with --by-benchmark")
        for benchmark in benchmark_names(args.base_dir):
            df = build_results_table(args.base_dir, simple=args.simple, benchmark=benchmark)
            print(f"Benchmark: {benchmark}")
            print(df.to_markdown(index=False, floatfmt=FLOAT_FORMAT, missingval=MISSING_VALUE))
        return

    df = build_results_table(args.base_dir, simple=args.simple)
    if args.csv:
        print(df.to_csv(index=False, float_format=f"%{FLOAT_FORMAT}", na_rep=MISSING_VALUE), end="")
        return

    output_path = args.output_path or args.base_dir / DEFAULT_OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format=f"%{FLOAT_FORMAT}", na_rep=MISSING_VALUE)
    print(f"Saved graph stats to {output_path}")
    print(df.to_markdown(index=False, floatfmt=FLOAT_FORMAT, missingval=MISSING_VALUE))


if __name__ == "__main__":
    main()
