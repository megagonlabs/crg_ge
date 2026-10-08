import argparse
import logging
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from tqdm import tqdm

from crg_ce.utils.openhands import load_conversation_state_and_events_from_archive

DATASET_PATH: str = "Brendan/openhands_ce_data_dev"
BASE_PATH: Path = Path("data")
N_WORKERS: int = 32

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate synced OpenHands conversation archives.")
    parser.add_argument("--dataset-path", default=DATASET_PATH)
    parser.add_argument("--base-path", type=Path, default=BASE_PATH)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    return parser.parse_args()


def get_conversation_archive_path(row: Mapping[str, Any]) -> Path:
    raw_path = row["conversation_archive_path"]
    if not isinstance(raw_path, str):
        raise TypeError(f"conversation_archive_path must be a string, got {type(raw_path)}")

    archive_path = Path(raw_path)
    if archive_path.is_absolute():
        raise ValueError(f"conversation_archive_path must be relative to the split output directory: {archive_path}")
    if ".." in archive_path.parts:
        raise ValueError(f"conversation_archive_path must not traverse upward: {archive_path}")
    return archive_path


def iter_split_rows(split: Dataset | IterableDataset) -> Iterator[Mapping[str, Any]]:
    for row in split:
        if not isinstance(row, dict):
            raise TypeError(f"Expected dataset row to be a dict, got {type(row)}")
        yield row


def validate_archive(*, row: Mapping[str, Any], output_dir: Path) -> Path:
    archive_path = output_dir / get_conversation_archive_path(row)
    state, events = load_conversation_state_and_events_from_archive(archive_path)
    if state is None:
        raise ValueError(f"Loaded archive without a conversation state: {archive_path}")
    if not events:
        raise ValueError(f"Loaded archive without events: {archive_path}")
    return archive_path


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    dataset = load_dataset(args.dataset_path)
    if not isinstance(dataset, DatasetDict | IterableDatasetDict):
        raise TypeError(f"Expected a dataset dict with splits, got {type(dataset)}")

    total_validated = 0
    for split_name, split in dataset.items():
        output_dir = args.base_path / args.dataset_path / str(split_name)
        rows = iter_split_rows(split)
        total = len(split) if isinstance(split, Dataset) else None

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(
                lambda row, output_dir=output_dir: validate_archive(row=row, output_dir=output_dir),
                rows,
            )
            validated_paths = list(tqdm(results, total=total, desc=f"validate {split_name}"))

        total_validated += len(validated_paths)
        logger.info("Validated %d archives for split %s", len(validated_paths), split_name)

    logger.info("Validated %d archives total", total_validated)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
