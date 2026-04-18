"""Generate a small synthetic Parquet dataset for the examples.

The dataset is intentionally tiny and split across multiple files so that
`FileShuffleConfig` has something meaningful to permute.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def generate_dataset(
    out_dir: Path,
    *,
    num_files: int = 8,
    rows_per_file: int = 250,
    overwrite: bool = True,
) -> Path:
    """Generate ``num_files`` Parquet files under ``out_dir``.

    Each row has a globally-unique ``id`` so downstream tests can reason about
    set membership independently of the deterministic ``row_hash`` column that
    `read_parquet(include_row_hash=True)` will synthesize.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            return out_dir
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    next_id = 0
    for file_idx in range(num_files):
        ids = np.arange(next_id, next_id + rows_per_file, dtype=np.int64)
        next_id += rows_per_file
        values = rng.standard_normal(rows_per_file).astype(np.float32)
        table = pa.table({"id": ids, "value": values})
        pq.write_table(table, out_dir / f"part-{file_idx:04d}.parquet")
    return out_dir


def default_data_dir() -> Path:
    """Default fixture path, scoped to this repo."""
    return Path(__file__).resolve().parents[2] / "data" / "fixture"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=default_data_dir())
    parser.add_argument("--num-files", type=int, default=8)
    parser.add_argument("--rows-per-file", type=int, default=250)
    args = parser.parse_args()
    path = generate_dataset(
        args.out_dir,
        num_files=args.num_files,
        rows_per_file=args.rows_per_file,
    )
    print(f"Wrote {args.num_files} files ({args.num_files * args.rows_per_file} rows) to {path}")


if __name__ == "__main__":
    main()
