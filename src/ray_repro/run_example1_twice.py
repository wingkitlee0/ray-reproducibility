"""Driver: run ``example1_reproducibility`` twice in subprocesses and assert
that the two runs produce identical per-epoch fingerprints.

Each invocation is a fresh Python process with a fresh Ray cluster, so any
cross-run match must come from seeded pipeline behavior rather than
in-process caching.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .fixture import default_data_dir, generate_dataset


def _run_once(
    seed: int,
    epochs: int,
    data_dir: Path,
    batch_size: int,
    shuffle_buffer: int,
    num_cpus: int,
) -> list[str]:
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
        "--json",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"Subprocess exited with code {proc.returncode}")

    # The last non-empty stdout line is the JSON payload.
    json_line = next(
        line for line in reversed(proc.stdout.splitlines()) if line.strip()
    )
    payload = json.loads(json_line)
    return payload["fingerprints"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--num-cpus", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        generate_dataset(args.data_dir)

    print(f"[run A] seed={args.seed} epochs={args.epochs}")
    fps_a = _run_once(
        args.seed,
        args.epochs,
        args.data_dir,
        args.batch_size,
        args.shuffle_buffer,
        args.num_cpus,
    )
    print(f"[run B] seed={args.seed} epochs={args.epochs}")
    fps_b = _run_once(
        args.seed,
        args.epochs,
        args.data_dir,
        args.batch_size,
        args.shuffle_buffer,
        args.num_cpus,
    )

    print()
    print("epoch | run A                                | run B                                | match")
    all_match = True
    for epoch, (a, b) in enumerate(zip(fps_a, fps_b)):
        match = a == b
        all_match = all_match and match
        print(f"{epoch:>5} | {a:<38} | {b:<38} | {match}")

    print()
    # Also check intra-run variation: epochs within one run should differ.
    unique_within_run = len(set(fps_a)) == len(fps_a)
    print(f"Reproducibility across runs: {'OK' if all_match else 'FAILED'}")
    print(
        f"Intra-run variation (reseed per epoch): {'OK' if unique_within_run else 'FAILED'}"
    )

    if not all_match or not unique_within_run:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
