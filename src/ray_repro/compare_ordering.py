"""Measure how parallelism perturbs row ordering, on a continuous scale.

The seeded pipeline is executed in fresh subprocesses at each requested
``--num-cpus`` value. The ``num_cpus=1`` run is taken as the reference
(perfect, naturally deterministic) order; every other run is compared
against it per epoch with three metrics:

- ``exact_match_fraction`` -- fraction of rows that land in the exact same
  position as the reference.
- ``spearman_rho``         -- Spearman rank correlation. 1.0 = identical
  order, ~0.0 = uncorrelated, -1.0 = reversed.
- ``displacement_score``   -- ``1 - mean(|Δrank|) / E_random``, clamped to
  ``[0, 1]``. 1.0 = identical; 0.0 means mean positional shift matches a
  uniformly random permutation. Most aligned with the intuition that "more
  CPUs perturb ordering locally but don't teleport rows across the epoch".

Typical output shape: scores decrease monotonically (on average) as
``num_cpus`` grows, but remain well above 0 because workers continue to
process rows that are *near* their reference positions -- they don't
randomize globally.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from argparse import BooleanOptionalAction
from pathlib import Path

import numpy as np

from .common import ordering_metrics
from .fixture import default_data_dir, generate_dataset


def _run_example1(
    *,
    seed: int,
    epochs: int,
    data_dir: Path,
    batch_size: int,
    shuffle_buffer: int,
    num_cpus: int,
    sequences_dir: Path,
    file_shuffle: bool,
    randomize_block_order: bool,
) -> None:
    """Invoke the example in a fresh subprocess so each Ray cluster is independent."""
    cmd = [
        sys.executable,
        "-m",
        "ray_repro.example1_reproducibility",
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--shuffle-buffer",
        str(shuffle_buffer),
        "--num-cpus",
        str(num_cpus),
        "--data-dir",
        str(data_dir),
        "--sequences-dir",
        str(sequences_dir),
    ]
    cmd.append("--file-shuffle" if file_shuffle else "--no-file-shuffle")
    cmd.append(
        "--randomize-block-order"
        if randomize_block_order
        else "--no-randomize-block-order"
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"example1 subprocess exited with code {proc.returncode} "
            f"(num_cpus={num_cpus})"
        )


def _load_sequences(sequences_dir: Path, epochs: int) -> list[np.ndarray]:
    return [np.load(sequences_dir / f"epoch_{e}.npy") for e in range(epochs)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument(
        "--num-cpus",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="CPU counts to sweep. The minimum value is used as the reference.",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--keep-sequences",
        type=Path,
        default=None,
        help="If provided, copy dumped sequences under this directory instead "
             "of discarding them when the driver exits.",
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
            "Add the all-to-all randomize_block_order() stage (default: off). "
            "Enabling it collapses scores to ~0 for any parallelism because "
            "the seeded shuffle runs on a parallelism-dependent arrival order."
        ),
    )
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        generate_dataset(args.data_dir)

    num_cpus_list = sorted(set(args.num_cpus))
    reference_nc = num_cpus_list[0]

    with tempfile.TemporaryDirectory(prefix="ray_repro_seqs_") as tmp:
        tmp_path = Path(tmp)
        sequences_by_nc: dict[int, list[np.ndarray]] = {}

        for nc in num_cpus_list:
            seq_dir = tmp_path / f"ncpus_{nc}"
            print(f"[sweep] running num_cpus={nc} ...", flush=True)
            _run_example1(
                seed=args.seed,
                epochs=args.epochs,
                data_dir=args.data_dir,
                batch_size=args.batch_size,
                shuffle_buffer=args.shuffle_buffer,
                num_cpus=nc,
                sequences_dir=seq_dir,
                file_shuffle=args.file_shuffle,
                randomize_block_order=args.randomize_block_order,
            )
            sequences_by_nc[nc] = _load_sequences(seq_dir, args.epochs)

        if args.keep_sequences is not None:
            args.keep_sequences.mkdir(parents=True, exist_ok=True)
            for nc, seqs in sequences_by_nc.items():
                out_dir = args.keep_sequences / f"ncpus_{nc}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for e, seq in enumerate(seqs):
                    np.save(out_dir / f"epoch_{e}.npy", seq)

        ref_seqs = sequences_by_nc[reference_nc]

    # Report ---------------------------------------------------------------
    print()
    print(f"Reference: num_cpus={reference_nc}  (score 1.000 by definition)")
    print()
    header = (
        f"{'num_cpus':>8} | {'epoch':>5} | {'n':>5} | "
        f"{'exact':>7} | {'spearman':>8} | {'disp_score':>10} | {'mean|Δ|':>8}"
    )
    print(header)
    print("-" * len(header))

    aggregates: dict[int, list[dict]] = {nc: [] for nc in num_cpus_list}
    for nc in num_cpus_list:
        for e, (obs, ref) in enumerate(zip(sequences_by_nc[nc], ref_seqs)):
            m = ordering_metrics(obs, ref)
            aggregates[nc].append(m)
            print(
                f"{nc:>8} | {e:>5} | {m['n']:>5} | "
                f"{m['exact_match_fraction']:>7.3f} | "
                f"{m['spearman_rho']:>8.3f} | "
                f"{m['displacement_score']:>10.3f} | "
                f"{m['mean_rank_displacement']:>8.2f}"
            )

    print()
    print("per-num_cpus averages across epochs:")
    print(f"{'num_cpus':>8} | {'exact':>7} | {'spearman':>8} | {'disp_score':>10}")
    print("-" * 46)
    for nc in num_cpus_list:
        ms = aggregates[nc]
        avg_exact = sum(m["exact_match_fraction"] for m in ms) / len(ms)
        avg_rho = sum(m["spearman_rho"] for m in ms) / len(ms)
        avg_disp = sum(m["displacement_score"] for m in ms) / len(ms)
        print(
            f"{nc:>8} | {avg_exact:>7.3f} | {avg_rho:>8.3f} | {avg_disp:>10.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
