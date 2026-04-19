"""Minimal demo for PR 2 (_execution_idx fix).

Rebuilds a fresh Dataset each epoch using only `read_parquet(shuffle=FileShuffleConfig(...))`.
NO iter_batches local shuffle buffer, NO randomize_block_order, NO random_shuffle.

With `reseed_after_execution=True`, each epoch should see a DIFFERENT file order.
On master this fails because `DataContext.copy()` gives every new Dataset a fresh
copy of the still-zero global `_execution_idx`.
"""
from __future__ import annotations

import ray
from ray.data import DataContext
from ray.data.datasource import FileShuffleConfig

from .fixture import default_data_dir, generate_dataset


def first_file_of_epoch(data_dir: str, seed: int, epoch: int) -> str:
    ds = ray.data.read_parquet(
        data_dir,
        shuffle=FileShuffleConfig(seed=seed, reseed_after_execution=True),
        include_paths=True,
    )
    # Take one row, look at its source file. No shuffling downstream.
    row = ds.take(1)[0]
    return row["path"].rsplit("/", 1)[-1]


def main() -> None:
    data_dir = default_data_dir()
    generate_dataset(data_dir)
    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
    try:
        ctx = DataContext.get_current()
        print(f"global _execution_idx at start: {ctx._execution_idx}")
        for epoch in range(4):
            f = first_file_of_epoch(data_dir, seed=42, epoch=epoch)
            print(
                f"epoch {epoch}: first_file={f!s:<12}  "
                f"global _execution_idx={ctx._execution_idx}"
            )
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
