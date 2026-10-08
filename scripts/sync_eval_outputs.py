import argparse
import logging
import shutil
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict, load_dataset
from tqdm import tqdm

DATASET_PATH: str = "Brendan/openhands_ce_data_dev"
BASE_PATH: Path = Path("data")
DEFAULT_OUTPUTS_BASE_PATH: Path = BASE_PATH / "eval_outputs"
N_WORKERS: int = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy trajectory archives into the dataset's local data directory.")
    parser.add_argument("--dataset-path", default=DATASET_PATH)
    parser.add_argument("--base-path", type=Path, default=BASE_PATH)
    parser.add_argument("--traj-outputs-path", type=Path, default=DEFAULT_OUTPUTS_BASE_PATH)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--force_redownload", action="store_true", default=False)
    return parser.parse_args()


def get_conversation_archive_path(row: Mapping[str, Any]) -> Path:
    raw_path = row["conversation_archive_path"]
    if not isinstance(raw_path, str):
        raise TypeError(f"conversation_archive_path must be a string, got {type(raw_path)}")

    archive_path = Path(raw_path)
    if archive_path.is_absolute():
        raise ValueError(f"conversation_archive_path must be relative to TRAJ_OUTPUTS_PATH: {archive_path}")
    if ".." in archive_path.parts:
        raise ValueError(f"conversation_archive_path must not traverse upward: {archive_path}")
    return archive_path


def sync_archive(*, row: Mapping[str, Any], output_dir: Path, traj_outputs_path: Path, force_redownload: bool) -> Path:
    archive_path = get_conversation_archive_path(row)
    source_path = traj_outputs_path / archive_path
    local_path = output_dir / archive_path
    local_path.parent.mkdir(exist_ok=True, parents=True)
    if force_redownload or not local_path.exists():
        shutil.copy2(source_path, local_path)
    else:
        logging.info(f"Skipping {archive_path}: already present (use --force_redownload to force overwrite)")
    return local_path


def iter_split_rows(split: Dataset | IterableDataset) -> Iterator[Mapping[str, Any]]:
    for row in split:
        if not isinstance(row, dict):
            raise TypeError(f"Expected dataset row to be a dict, got {type(row)}")
        yield row


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError(f"--workers must be >= 1, got {args.workers}")

    dataset = load_dataset(args.dataset_path)
    if not isinstance(dataset, DatasetDict | IterableDatasetDict):
        raise TypeError(f"Expected a dataset dict with splits, got {type(dataset)}")

    for split_name, split in dataset.items():
        output_dir = args.base_path / args.dataset_path / str(split_name)
        output_dir.mkdir(exist_ok=True, parents=True)
        rows = iter_split_rows(split)
        total = len(split) if isinstance(split, Dataset) else None
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(
                lambda row, output_dir=output_dir: sync_archive(
                    row=row,
                    output_dir=output_dir,
                    traj_outputs_path=args.traj_outputs_path,
                    force_redownload=args.force_redownload,
                ),
                rows,
            )
            list(tqdm(results, total=total, desc=f"sync {split_name}"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
