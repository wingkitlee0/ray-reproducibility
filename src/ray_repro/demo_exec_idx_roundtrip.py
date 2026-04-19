"""Test: on *master* (no PR 2), can the user round-trip _execution_idx themselves?

Simulates checkpoint/resume by manually setting the global _execution_idx before
each rebuild, then reading the final value off ds._plan._context after iteration.
"""
from __future__ import annotations

import ray
from ray.data import DataContext
from ray.data.datasource import FileShuffleConfig

from .fixture import default_data_dir, generate_dataset


def one_epoch(data_dir: str, seed: int, starting_idx: int) -> tuple[str, int]:
    DataContext.get_current()._execution_idx = starting_idx
    ds = ray.data.read_parquet(
        data_dir,
        shuffle=FileShuffleConfig(seed=seed, reseed_after_execution=True),
        include_paths=True,
    )
    before = ds._plan._context._execution_idx
    first_file = ds.take(1)[0]["path"].rsplit("/", 1)[-1]
    # Force full execution so the after-execution callback fires
    _ = ds.count()
    after = ds._plan._context._execution_idx
    return first_file, before, after


def main() -> None:
    data_dir = default_data_dir()
    generate_dataset(data_dir)
    ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level="ERROR")
    try:
        saved_idx = 0
        for epoch in range(4):
            first_file, before, after = one_epoch(data_dir, seed=42, starting_idx=saved_idx)
            print(
                f"epoch {epoch}: start_idx={saved_idx}  ds.ctx before={before}  "
                f"first_file={first_file}  ds.ctx after={after}"
            )
            saved_idx = after
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
