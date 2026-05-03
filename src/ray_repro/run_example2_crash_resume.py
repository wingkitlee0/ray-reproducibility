"""Driver: simulate a crash mid-epoch and a resume in a second process.

Runs ``example2_checkpoint`` twice against a shared store root:

1. **First subprocess** clears the store (``--reset``) and runs with
   ``--crash-after EPOCH:BATCHES``. It completes every epoch before
   ``EPOCH`` to disk, consumes ``BATCHES`` batches of ``EPOCH``, persists
   the partial segment, and exits with code 1.
2. **Second subprocess** starts with no ``--reset`` and no
   ``--crash-after``. It reads the store, restores ``_execution_idx``,
   discovers that ``EPOCH`` has a non-empty set of seen hashes, filters
   those rows out of the rebuilt pipeline, drains the remainder, then
   runs every remaining epoch normally.

The driver then verifies:

- The crash happened where requested and wrote the expected partial
  segment.
- The resume picked up from the recorded ``current_epoch``.
- Each epoch's row-hash segments are disjoint and their union covers
  every fixture row exactly once.
- The final ``current_epoch`` / ``execution_idx`` reflect all epochs
  completed.

No torch model, Ray Train, or streaming_split -- just a realistic-shaped
crash/resume test of the checkpoint protocol.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from .common import default_data_dir
from .fixture import generate_dataset


def _run_subprocess(
    *,
    total_epochs: int,
    seed: int,
    batch_size: int,
    shuffle_buffer: int,
    data_dir: Path,
    store_root: Path,
    crash_after: str | None,
    reset: bool,
    label: str,
) -> dict:
    cmd = [
        sys.executable,
        "-m",
        "ray_repro.example2_checkpoint",
        "--seed",
        str(seed),
        "--total-epochs",
        str(total_epochs),
        "--batch-size",
        str(batch_size),
        "--shuffle-buffer",
        str(shuffle_buffer),
        "--data-dir",
        str(data_dir),
        "--store-root",
        str(store_root),
        "--json",
    ]
    if crash_after:
        cmd += ["--crash-after", crash_after]
    if reset:
        cmd += ["--reset"]

    print(f"[driver] {label}: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    # Stream both streams verbatim so the crash/resume narrative is visible
    # even when something unexpected fails.
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)

    json_line = next(
        (
            line
            for line in reversed(proc.stdout.splitlines())
            if line.strip().startswith("{")
        ),
        None,
    )
    if json_line is None:
        raise SystemExit(
            f"{label}: subprocess produced no JSON output (exit={proc.returncode})"
        )

    return {"exit": proc.returncode, "payload": json.loads(json_line)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-epochs", type=int, default=3)
    parser.add_argument(
        "--crash-after",
        type=str,
        default="1:10",
        help="EPOCH:BATCHES for the simulated crash in the first subprocess.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args(argv)

    if not args.data_dir.exists():
        generate_dataset(args.data_dir)

    repo_root = Path(__file__).resolve().parents[2]
    store_root = repo_root / "data" / "seen_hashes_crash_resume"
    if store_root.exists():
        shutil.rmtree(store_root)

    # --- first subprocess: crash ---
    first = _run_subprocess(
        total_epochs=args.total_epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        data_dir=args.data_dir,
        store_root=store_root,
        crash_after=args.crash_after,
        reset=True,
        label="run 1 (crash)",
    )
    print(file=sys.stderr)

    crash_epoch, crash_batches = (int(x) for x in args.crash_after.split(":"))

    if not first["payload"].get("crashed"):
        print(
            f"[driver] FAIL: run 1 did not crash as expected "
            f"(exit={first['exit']}, payload={first['payload']})",
            file=sys.stderr,
        )
        return 1
    if first["exit"] != 1:
        print(
            f"[driver] FAIL: run 1 expected exit=1, got {first['exit']}",
            file=sys.stderr,
        )
        return 1
    if first["payload"]["crash_epoch"] != crash_epoch:
        print(
            f"[driver] FAIL: run 1 crashed on wrong epoch "
            f"(expected {crash_epoch}, got {first['payload']['crash_epoch']})",
            file=sys.stderr,
        )
        return 1
    if first["payload"]["crash_batches"] != crash_batches:
        print(
            f"[driver] FAIL: run 1 crashed at wrong batch count "
            f"(expected {crash_batches}, got {first['payload']['crash_batches']})",
            file=sys.stderr,
        )
        return 1
    if first["payload"]["final_current_epoch"] != crash_epoch:
        print(
            f"[driver] FAIL: run 1 final current_epoch should be {crash_epoch}, "
            f"got {first['payload']['final_current_epoch']}",
            file=sys.stderr,
        )
        return 1

    # --- second subprocess: resume ---
    second = _run_subprocess(
        total_epochs=args.total_epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        shuffle_buffer=args.shuffle_buffer,
        data_dir=args.data_dir,
        store_root=store_root,
        crash_after=None,
        reset=False,
        label="run 2 (resume)",
    )
    print(file=sys.stderr)

    if second["exit"] != 0:
        print(
            f"[driver] FAIL: run 2 expected clean exit, got {second['exit']}",
            file=sys.stderr,
        )
        return 1

    resumed_epochs = second["payload"]["epochs"]
    expected_resumed_epoch_ids = list(range(crash_epoch, args.total_epochs))
    got_resumed_epoch_ids = [e["epoch"] for e in resumed_epochs]
    if got_resumed_epoch_ids != expected_resumed_epoch_ids:
        print(
            f"[driver] FAIL: run 2 processed epochs {got_resumed_epoch_ids}, "
            f"expected {expected_resumed_epoch_ids}",
            file=sys.stderr,
        )
        return 1

    resumed_epoch_result = next(e for e in resumed_epochs if e["epoch"] == crash_epoch)
    if resumed_epoch_result["segments"] != 2:
        print(
            f"[driver] FAIL: resumed epoch {crash_epoch} should have 2 segments "
            f"(one from crash, one from resume), got "
            f"{resumed_epoch_result['segments']}",
            file=sys.stderr,
        )
        return 1

    all_invariants_ok = all(
        e["segments_disjoint"] and e["union_equals_expected"] for e in resumed_epochs
    )
    if not all_invariants_ok:
        print(
            "[driver] FAIL: one or more resumed epochs failed the "
            "disjoint/union invariants",
            file=sys.stderr,
        )
        return 1

    final_epoch = second["payload"]["final_current_epoch"]
    final_idx = second["payload"]["final_execution_idx"]
    if final_epoch != args.total_epochs:
        print(
            f"[driver] FAIL: final current_epoch should be {args.total_epochs}, "
            f"got {final_epoch}",
            file=sys.stderr,
        )
        return 1
    if final_idx != args.total_epochs:
        print(
            f"[driver] FAIL: final execution_idx should be {args.total_epochs} "
            f"(one per completed epoch), got {final_idx}",
            file=sys.stderr,
        )
        return 1

    print()
    print("=" * 70)
    print(f"Crash at epoch {crash_epoch} after {crash_batches} batches: OK")
    print(
        f"Resume picked up at epoch {crash_epoch} and completed through "
        f"epoch {args.total_epochs - 1}: OK"
    )
    print(
        f"Resumed epoch {crash_epoch} has 2 segments "
        f"(crash partial + resume remainder), "
        f"{resumed_epoch_result['rows_consumed']}/"
        f"{resumed_epoch_result['expected_rows']} rows, "
        f"disjoint + union OK"
    )
    print(f"Final store state: current_epoch={final_epoch} execution_idx={final_idx}")
    print("All crash/resume invariants: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
