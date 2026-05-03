import ray
from ray.data import SaveMode
from ray.data.expressions import random
from pathlib import Path
from typing import Optional

from ray.data.datasource import FilenameProvider

from ray_repro.fixture import DEFAULT_RANDOM_SEED


class SimpleFilenameProvider(FilenameProvider):

    def __init__(self, prefix: str, file_format: str):
        super().__init__(file_format=file_format)
        self.prefix = prefix

    def get_filename_for_task(self, write_uuid, task_index):
        # return f"{self.prefix}_{write_uuid}_{task_index:06}.{self.file_format}"
        return f"{self.prefix}_{task_index:06}.{self.file_format}"


def generate_dataset_with_ray(
    out_dir: Path,
    *,
    num_files: int = 8,
    rows_per_file: int = 250,
    overwrite: bool = True,
    seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> Path:
    """Generate ``num_files`` Parquet files under ``out_dir`` using Ray Data."""
    

    ds = (
        ray.data.range(num_files * rows_per_file)
        .with_column("value", random(seed=seed))
        .repartition(num_files)
    )

    ds.write_parquet(
        out_dir,
        filename_provider=SimpleFilenameProvider(prefix="part", file_format="parquet"),
        mode=SaveMode.OVERWRITE if overwrite else SaveMode.ERROR,
    )
    return out_dir