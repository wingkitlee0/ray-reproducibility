"""Example 1: demonstrate deterministic Ray Data ingestion across runs.

The script iterates a seeded pipeline for ``--epochs`` epochs and prints one
order-dependent fingerprint per epoch. Run the script twice with the same
``--seed`` and the per-epoch fingerprints must match byte-for-byte.

Within a single run, fingerprints across epochs differ because
``RandomSeedConfig(reseed_after_execution=True)`` advances the seed per
pipeline execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import ray

from .common import PipelineConfig, default_data_dir, run_epochs
from .fixture import generate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=1,
        help=(
            "CPUs exposed to the Ray cluster. Default is 1 so task-completion "
            "order is naturally deterministic without preserve_order."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Path to the Parquet fixture. Created on first run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit fingerprints as a JSON object on the final line (easier for drivers to parse).",
    )
    parser.add_argument(
        "--sequences-dir",
        type=Path,
        default=None,
        help=(
            "If set, dump each epoch's row_hash sequence as epoch_{i}.npy "
            "under this directory. Used by the compare_ordering driver."
        ),
    )
    parser.add_argument(
        "--file-shuffle",
        action=BooleanOptionalAction,
        default=True,
        help="Use FileShuffleConfig in read_parquet (default: on).",
    )
    parser.add_argument(
        "--randomize-block-order",
        action=BooleanOptionalAction,
        default=False,
        help=(
            "Add the AllToAll randomize_block_order() stage (default: off). "
            "Its seeded permutation runs over a parallelism-dependent block "
            "arrival order, which collapses ordering metrics to ~0 unless "
            "preserve_order is also enabled."
        ),
    )
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        print(f"Fixture not found at {args.data_dir}; generating...", file=sys.stderr)
        generate_dataset(args.data_dir)

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        local_shuffle_buffer_size=args.shuffle_buffer,
        enable_file_shuffle=args.file_shuffle,
        enable_randomize_block_order=args.randomize_block_order,
    )

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True, log_to_driver=False)
    try:
        fingerprints = run_epochs(cfg, args.epochs, sequences_dir=args.sequences_dir)
    finally:
        ray.shutdown()

    for epoch, fp in enumerate(fingerprints):
        print(f"epoch={epoch} fingerprint={fp}")
    if args.json:
        print(json.dumps({"seed": args.seed, "fingerprints": fingerprints}))

    # Sanity check: fingerprints across epochs should differ when
    # reseed_after_execution is enabled.
    unique = set(fingerprints)
    if len(unique) != len(fingerprints):
        print(
            "WARNING: some epochs produced identical fingerprints -- "
            "reseed_after_execution may not be taking effect.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
