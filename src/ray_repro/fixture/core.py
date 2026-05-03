import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


from . import DEFAULT_RANDOM_SEED


def generate_dataset(
    out_dir: Path,
    *,
    num_files: int = 8,
    rows_per_file: int = 250,
    overwrite: bool = True,
    seed: Optional[int] = DEFAULT_RANDOM_SEED,
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
