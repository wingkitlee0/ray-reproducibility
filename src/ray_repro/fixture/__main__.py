"""Generate a small synthetic Parquet dataset for the examples.

The dataset is intentionally tiny and split across multiple files so that
`FileShuffleConfig` has something meaningful to permute.
"""

from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path

from ..common import default_data_dir
from . import DEFAULT_RANDOM_SEED, generate_dataset, generate_dataset_with_ray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=default_data_dir())
    parser.add_argument("--num-files", type=int, default=8)
    parser.add_argument("--rows-per-file", type=int, default=250)
    parser.add_argument(
        "-s",
        "--seed",
        nargs="?",
        type=int,
        const=DEFAULT_RANDOM_SEED,
        help=f"random seed. If no flag is provided, the seed is None. If -s/--seed is provided without a value, it will use the default value {DEFAULT_RANDOM_SEED}.",
    )
    parser.add_argument("--use-ray", action="store_true")
    parser.add_argument(
        "--overwrite",
        action=BooleanOptionalAction,
        default=True,
        help="Replace existing fixture files. With --no-overwrite, fail if the output dir already exists.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"{args=!r}")

    if args.dry_run:
        print("Dry run, exiting...")
        return

    if args.use_ray:
        path = generate_dataset_with_ray(
            args.out_dir,
            num_files=args.num_files,
            rows_per_file=args.rows_per_file,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    else:
        path = generate_dataset(
            args.out_dir,
            num_files=args.num_files,
            rows_per_file=args.rows_per_file,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    print(
        f"Wrote {args.num_files} files ({args.num_files * args.rows_per_file} rows) to {path}"
    )


if __name__ == "__main__":
    main()
