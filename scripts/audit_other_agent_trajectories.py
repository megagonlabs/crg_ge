import argparse
import json
import logging
import re
import traceback
from collections import Counter
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from openhands.sdk.event import ACPToolCallEvent, ActionEvent
from openhands.sdk.tool.builtins.finish import FinishAction
from tqdm import tqdm

from crg_ce.utils.openhands import load_conversation_state_and_events_from_archive
from crg_ce.utils.openhands_trajectory import (
    _extract_recognized_acp_content,
    condense_uncited_action_steps,
    get_active_branch_events,
    render_trajectory,
)

DATASET_PATH = "Brendan/openhands_ce_data_swe_bench_other_agents"
BASE_PATH = Path("data")
N_WORKERS = 8
_ACTION_TAG = re.compile(r"^<action step=", re.MULTILINE)
_OBSERVATION_TAG = re.compile(r"^<observation step=", re.MULTILINE)
_SUMMARIZED_ACTION_TAG = re.compile(r"^<summarized_action step=", re.MULTILINE)
_SUMMARIZED_OBSERVATION_TAG = re.compile(r"^<summarized_observation step=", re.MULTILINE)


@dataclass(frozen=True)
class AuditResult:
    split: str
    row_index: int
    instance_id: str
    model: str
    archive_path: str
    status: str
    action_count: int = 0
    observation_count: int = 0
    warnings: tuple[str, ...] = ()
    error: str | None = None
    traceback: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, render, condense, and audit Claude Code/Codex OpenHands trajectories."
    )
    parser.add_argument("--dataset-path", default=DATASET_PATH)
    parser.add_argument("--name", default=None, help="Optional Hugging Face dataset config name.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--base-path", type=Path, default=BASE_PATH)
    parser.add_argument("--split", action="append", help="Split to audit; repeat for multiple splits. Defaults to all.")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to audit per split.")
    parser.add_argument("--report-path", type=Path, default=Path("trajectory_render_audit.json"))
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser.parse_args()


def _required_nonblank_string(row: Mapping[str, Any], field: str) -> str:
    value = row[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string, got {value!r}")
    return value


def _archive_path(row: Mapping[str, Any], split_root: Path) -> Path:
    relative_path = Path(_required_nonblank_string(row, "conversation_archive_path"))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"conversation_archive_path must be a safe relative path, got {relative_path}")
    archive_path = split_root / relative_path
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    return archive_path


def _row_warnings(row: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    benchmark = row.get("benchmark")
    if not isinstance(benchmark, str) or not benchmark.strip():
        warnings.append("missing nonblank benchmark; domain-specific confidence prompts may be incomplete")
    if "resolved" not in row or not isinstance(row["resolved"], bool):
        warnings.append("missing boolean resolved label; confidence estimates cannot be evaluated for calibration")
    trajectory_type = row.get("trajectory_type")
    if trajectory_type is not None and trajectory_type != "acp":
        warnings.append(
            f"unexpected trajectory_type={trajectory_type!r}; expected null for archives or 'acp' for JSONL"
        )
    return warnings


def _acp_warnings(events: list[Any]) -> list[str]:
    warnings: list[str] = []
    calls: dict[str, list[ACPToolCallEvent]] = {}
    for event in events:
        if isinstance(event, ACPToolCallEvent):
            calls.setdefault(event.tool_call_id, []).append(event)

    fallback_content_calls = 0
    for snapshots in calls.values():
        terminal = next((event for event in reversed(snapshots) if event.status in {"completed", "failed"}), None)
        if terminal is None:
            continue
        if terminal.content and _extract_recognized_acp_content(terminal.content) is None:
            fallback_content_calls += 1

    if fallback_content_calls:
        warnings.append(
            f"{fallback_content_calls} ACP observations use lossless JSON fallback because their content shape is "
            "unknown"
        )
    return warnings


def audit_row(*, row: Mapping[str, Any], split: str, row_index: int, split_root: Path) -> AuditResult:
    instance_id = str(row.get("instance_id", "<missing>"))
    model = str(row.get("model", "<missing>"))
    archive_path = str(split_root / str(row.get("conversation_archive_path", "<missing>")))
    action_count = 0
    observation_count = 0
    try:
        instance_id = _required_nonblank_string(row, "instance_id")
        model = _required_nonblank_string(row, "model")
        _required_nonblank_string(row, "problem_statement")
        resolved_archive_path = _archive_path(row, split_root)
        archive_path = str(resolved_archive_path)
        trajectory_type = row.get("trajectory_type")
        if trajectory_type == "acp" and resolved_archive_path.name.endswith(".tar.gz"):
            raise ValueError(
                "trajectory_type='acp' is reserved for raw ACP JSONL, but conversation_archive_path points to a "
                ".tar.gz OpenHands archive; remove trajectory_type or set it to null"
            )
        state, events = load_conversation_state_and_events_from_archive(
            resolved_archive_path,
            trajectory_type=trajectory_type,
        )
        if not events:
            raise ValueError("archive contains no events")

        active_events = get_active_branch_events(events, state)
        if not any(isinstance(event, ActionEvent | ACPToolCallEvent) for event in active_events):
            raise ValueError("active trajectory contains no actions")

        full_rendering = render_trajectory(
            events,
            state=state,
            start_at_first_action_event=True,
        )
        action_count = len(_ACTION_TAG.findall(full_rendering))
        observation_count = len(_OBSERVATION_TAG.findall(full_rendering))
        if action_count == 0:
            raise ValueError("production rendering contains no action tags")
        if action_count != observation_count:
            raise ValueError(f"rendered action/observation count mismatch: {action_count} != {observation_count}")
        if "<message role=user>" in full_rendering:
            raise ValueError("production action-first rendering retained a user message")

        first_action_index = next(
            index for index, event in enumerate(active_events) if isinstance(event, ActionEvent | ACPToolCallEvent)
        )
        action_first_events = active_events[first_action_index:]
        condensed_events = condense_uncited_action_steps(action_first_events, cited_step_numbers=set())
        condensed_rendering = render_trajectory(
            condensed_events,
            step_number_source_events=action_first_events,
        )
        if len(_SUMMARIZED_ACTION_TAG.findall(condensed_rendering)) != action_count:
            raise ValueError("fully condensed rendering does not summarize every rendered action")
        if len(_SUMMARIZED_OBSERVATION_TAG.findall(condensed_rendering)) != observation_count:
            raise ValueError("fully condensed rendering does not summarize every rendered observation")

        warnings = _row_warnings(row)
        warnings.extend(_acp_warnings(active_events))
        finish_count = sum(
            isinstance(event, ActionEvent) and isinstance(event.action, FinishAction) for event in active_events
        )
        if finish_count != 1:
            warnings.append(f"expected one FinishAction carrying combined agent messages, found {finish_count}")
        if full_rendering.count("<combined_agent_messages>") != finish_count:
            raise ValueError("combined-agent-message blocks do not match FinishAction count")

        return AuditResult(
            split=split,
            row_index=row_index,
            instance_id=instance_id,
            model=model,
            archive_path=archive_path,
            status="warning" if warnings else "ok",
            action_count=action_count,
            observation_count=observation_count,
            warnings=tuple(warnings),
        )
    except Exception as error:
        return AuditResult(
            split=split,
            row_index=row_index,
            instance_id=instance_id,
            model=model,
            archive_path=archive_path,
            status="error",
            action_count=action_count,
            observation_count=observation_count,
            error=f"{type(error).__name__}: {error}",
            traceback="".join(traceback.format_exception(type(error), error, error.__traceback__)),
        )


def _audit_rows(
    rows: list[Mapping[str, Any]],
    *,
    split: str,
    split_root: Path,
    workers: int,
) -> list[AuditResult]:
    tasks = [dict(row=row, split=split, row_index=index, split_root=split_root) for index, row in enumerate(rows)]
    if workers == 1:
        return [audit_row(**task) for task in tqdm(tasks, desc=f"audit {split}")]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(lambda task: audit_row(**task), tasks),
                total=len(tasks),
                desc=f"audit {split}",
            )
        )


def _validate_unique_output_keys(rows_by_split: Mapping[str, list[Mapping[str, Any]]]) -> None:
    for split, rows in rows_by_split.items():
        keys = Counter((str(row.get("instance_id")), str(row.get("model"))) for row in rows)
        duplicates = [key for key, count in keys.items() if count > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate (instance_id, model) keys in {split} would overwrite confidence outputs: "
                + ", ".join(map(str, duplicates[:20]))
            )


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")
    if args.limit is not None and args.limit < 1:
        raise ValueError(f"--limit must be >= 1, got {args.limit}")

    loaded = load_dataset(args.dataset_path, name=args.name, revision=args.revision)
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(loaded)}")
    split_names = args.split or list(loaded)
    unknown_splits = set(split_names) - set(loaded)
    if unknown_splits:
        raise ValueError(f"Unknown splits: {sorted(unknown_splits)}")

    rows_by_split: dict[str, list[Mapping[str, Any]]] = {}
    for split in split_names:
        dataset = loaded[split]
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected materialized Dataset for {split}, got {type(dataset)}")
        selected = dataset if args.limit is None else dataset.select(range(min(args.limit, len(dataset))))
        rows_by_split[split] = list(selected)
    _validate_unique_output_keys(rows_by_split)

    results: list[AuditResult] = []
    for split, rows in rows_by_split.items():
        split_root = args.base_path / args.dataset_path / split
        results.extend(_audit_rows(rows, split=split, split_root=split_root, workers=args.workers))

    report = {
        "dataset_path": args.dataset_path,
        "counts": dict(Counter(result.status for result in results)),
        "results": [asdict(result) for result in results],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Audit counts: %s; report: %s", report["counts"], args.report_path)

    has_errors = any(result.status == "error" for result in results)
    has_warnings = any(result.status == "warning" for result in results)
    if has_errors or (args.fail_on_warning and has_warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
