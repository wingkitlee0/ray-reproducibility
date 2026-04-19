"""Example 2: data-pipeline checkpointing across process restarts.

Normally the script runs ``--total-epochs`` epochs of the seeded pipeline
to completion. At the end of every epoch it writes the consumed
``row_hash`` values plus the advanced ``_execution_idx`` counter to a
filesystem-backed :class:`SeenHashStore`, so a crashed process can resume
from exactly where it left off.

To demonstrate the mid-epoch case, pass ``--crash-after EPOCH:BATCHES``:

    # First invocation: finish epoch 0, consume 10 batches of epoch 1,
    # then exit(1) to simulate a crash.
    python -m ray_repro.example2_checkpoint \\
        --total-epochs 3 --reset --crash-after 1:10

    # Second invocation: the store already has one full epoch of hashes
    # and a partial segment under epoch 1. The script reads that state,
    # restores _execution_idx, filters the already-seen 640 rows out of
    # epoch 1, drains the remaining 1360, then runs epoch 2 normally.
    python -m ray_repro.example2_checkpoint --total-epochs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from argparse import BooleanOptionalAction
from pathlib import Path

import ray

from .checkpoint_store import SeenHashStore
from .common import CrashInjected, PipelineConfig, run_epochs_with_resume
from .fixture import default_data_dir, generate_dataset


def _default_store_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "seen_hashes"


def _parse_crash_after(value: str | None) -> tuple[int, int] | None:
    """Parse ``EPOCH:BATCHES`` into a ``(epoch, batches)`` tuple.

    ``None`` means "don't inject a crash"; an empty string means the same
    so argparse's default of an optional flag can be passed through
    unchanged from a wrapping driver.
    """
    if value is None or value == "":
        return None
    try:
        epoch_s, batches_s = value.split(":")
        return int(epoch_s), int(batches_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected EPOCH:BATCHES (e.g. '1:10'), got {value!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=3,
        help="Target number of epochs. If resuming, epochs already completed "
        "on disk are skipped and not re-run.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument(
        "--crash-after",
        type=str,
        default=None,
        help="Inject a simulated crash at EPOCH:BATCHES, e.g. '1:10' to "
        "exit(1) after consuming 10 batches of epoch 1.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the seen-hash store before starting (fresh run).",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=1,
        help="CPUs exposed to the Ray cluster. Default 1 for deterministic "
        "task-completion order.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Path to the Parquet fixture. Created on first run.",
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=_default_store_root(),
        help="Directory for the filesystem-backed seen-hash store.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the epoch results and final store state as JSON.",
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
        help="Add the AllToAll randomize_block_order() stage (default: off).",
    )
    args = parser.parse_args(argv)

    crash_at = _parse_crash_after(args.crash_after)

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

    store = SeenHashStore(args.store_root)
    if args.reset:
        store.clear_all()

    resume_epoch = store.load_current_epoch()
    resume_idx = store.load_execution_idx()
    print(
        f"[example2] store={args.store_root} current_epoch={resume_epoch} "
        f"execution_idx={resume_idx} target_epochs={args.total_epochs}",
        file=sys.stderr,
    )

    ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True, log_to_driver=False)
    crashed: CrashInjected | None = None
    try:
        try:
            results = run_epochs_with_resume(
                cfg,
                args.total_epochs,
                store,
                crash_at=crash_at,
            )
        except CrashInjected as exc:
            crashed = exc
            results = []
    finally:
        ray.shutdown()

    all_ok = True
    for r in results:
        ok = r.segments_disjoint and r.union_equals_expected
        all_ok = all_ok and ok
        print(
            f"epoch={r.epoch} segments={r.segments} "
            f"rows={r.rows_consumed}/{r.expected_rows} "
            f"disjoint={r.segments_disjoint} "
            f"union_ok={r.union_equals_expected} "
            f"fp={r.concat_fingerprint}"
        )

    final_epoch = store.load_current_epoch()
    final_idx = store.load_execution_idx()
    if crashed is not None:
        print(
            f"[example2] CRASH after {crashed.batches_consumed} batches "
            f"of epoch {crashed.epoch}. store now: current_epoch={final_epoch} "
            f"execution_idx={final_idx}",
            file=sys.stderr,
        )
    else:
        print(
            f"[example2] finished. store now: current_epoch={final_epoch} "
            f"execution_idx={final_idx}",
            file=sys.stderr,
        )

    if args.json:
        payload = {
            "seed": args.seed,
            "crashed": crashed is not None,
            "crash_epoch": crashed.epoch if crashed else None,
            "crash_batches": crashed.batches_consumed if crashed else None,
            "final_current_epoch": final_epoch,
            "final_execution_idx": final_idx,
            "epochs": [
                {
                    "epoch": r.epoch,
                    "segments": r.segments,
                    "rows_consumed": r.rows_consumed,
                    "expected_rows": r.expected_rows,
                    "segments_disjoint": r.segments_disjoint,
                    "union_equals_expected": r.union_equals_expected,
                    "concat_fingerprint": r.concat_fingerprint,
                }
                for r in results
            ],
        }
        print(json.dumps(payload))

    if crashed is not None:
        return 1
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
