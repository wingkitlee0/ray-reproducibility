# `_execution_idx` Semantics and the Checkpoint-Resume Contract

Design note captured while working on Example 2 (data-pipeline checkpointing)
against `wingkitlee0/ray@kit/repro-example`.

## TL;DR

- `DataContext._execution_idx` is **per Dataset**, by design.
- Two rebuilt datasets (`read_parquet(...)` called twice) are semantically
  independent pipelines. Each legitimately starts at `_execution_idx = 0`.
- The scenario where users *want* reseed behaviour to carry across rebuilds
  (training resume, crash recovery, per-epoch rebuilds for memory reasons) is a
  **checkpoint-resume** concern, not a Ray-internal concern.
- The correct pattern is: persist `_execution_idx` in the user's checkpoint
  alongside other resume state, and restore it into `DataContext` before
  rebuilding the pipeline.
- This means the `_ExecutionIdxCounter` / shared-counter fix I prototyped is
  **not needed**, and the `ShufflingBatcher` seed-pinning patch also becomes
  redundant.

## Background

Ray Data's reproducibility story exposes three seed-bearing surfaces:

1. `FileShuffleConfig(seed=..., reseed_after_execution=...)` for file ordering
   in `read_parquet`.
2. `RandomSeedConfig(...)` for global iterator-level reseeding.
3. `iter_batches(local_shuffle_seed=...)` for the per-iteration local shuffle
   buffer.

All three are routed through `DataContext._execution_idx`: a monotonic
per-execution counter combined with the user-supplied base seed so that
repeated executions of the same Dataset produce a *sequence* of reproducible
but distinct orderings.

When a Dataset is constructed, `ExecutionPlan` captures a *copy* of the current
`DataContext`:

```160:162:/home/wklee/github/ray/python/ray/data/read_api.py
execution_plan = ExecutionPlan(
    ...,
    DataContext.get_current().copy(),
```

From that point on the pipeline runs entirely against that per-Dataset copy;
`DataContext.get_current()` is not consulted during execution. When an
execution finishes, `ExecutionIdxUpdateCallback.after_execution_succeeds`
increments the *per-Dataset copy's* `_execution_idx`.

## The apparent bug (and why it isn't one)

The following two patterns behave differently, which initially looked like a
bug:

| Pattern | Behaviour | Intended? |
|---|---|---|
| `ds = read_parquet(...); for epoch in range(N): iterate(ds)` | `ds.ctx._execution_idx` goes 0 → 1 → 2. Reseed advances. | ✅ |
| `for epoch in range(N): ds = read_parquet(...); iterate(ds)` | Every new `ds.ctx` starts at 0. Reseed does **not** advance. | ✅ (see below) |

Under the per-Dataset model, the second pattern is **correct**: each `ds` is a
brand-new pipeline, unrelated to prior Datasets, and starts its own counter at
0. There is no globally shared "how many times has anything executed" notion,
and adding one would surprise users who build multiple independent pipelines
in the same driver (`ds1 = read_parquet(A); ds2 = read_parquet(B)`).

The user-visible surprise in the second pattern is purely a *documentation and
checkpoint* issue: the user wanted "continue the seed sequence across my
training job," but Ray has no way to know that these two `read_parquet` calls
are logically the same pipeline.

## The right solution: persist `_execution_idx` in the user checkpoint

Resume semantics belong in the user's checkpoint. The contract is:

1. Before building `ds`, set
   `DataContext.get_current()._execution_idx = <value loaded from checkpoint>`.
2. Build and execute `ds` as normal. The per-Dataset copy inherits the value,
   and `after_execution_succeeds` bumps it.
3. After iteration/execution, read the final value from
   `ds._plan._context._execution_idx` (internal today; could be promoted to a
   small public helper later).
4. Write that value back into the checkpoint alongside any other resume state
   (e.g. the set of already-seen `row_hash` values, in Example 2's case).

### Empirical confirmation on master

Minimal script (`src/ray_repro/demo_exec_idx_roundtrip.py`) on Ray master with
no patches at all:

```
epoch 0: start_idx=0  ds.ctx before=0  first_file=part-0007.parquet  ds.ctx after=2
epoch 1: start_idx=2  ds.ctx before=2  first_file=part-0006.parquet  ds.ctx after=4
epoch 2: start_idx=4  ds.ctx before=4  first_file=part-0006.parquet  ds.ctx after=6
epoch 3: start_idx=6  ds.ctx before=6  first_file=part-0002.parquet  ds.ctx after=8
```

The first file varies across epochs as soon as the user manages the counter.
No Ray-internal changes are required.

(The counter bumps by 2 per epoch because the demo triggers two executions per
epoch: one `take(1)` and one `count()`. This is a corollary: "epoch" as a user
concept maps to one-or-more Ray executions, and it is the user's job to pick
the unit they checkpoint.)

## Implications for the upstream PR plan

Original plan (three PRs):

1. `include_row_hash` for `read_parquet`.
2. `_execution_idx` fix (`_ExecutionIdxCounter`, shared-by-reference counter
   in `DataContext`).
3. `iter_batches` reproducibility + `ShufflingBatcher` seed pinning.

Revised plan (two PRs):

1. `include_row_hash` for `read_parquet` (unchanged, already in flight).
2. `iter_batches` reproducibility — `local_shuffle_seed` API, docs, tests.
   **Drop** the `ShufflingBatcher` seed-pinning patch: under per-Dataset
   semantics, `ds.ctx._execution_idx` only changes between executions, so
   reading it during compactions within a single `iter_batches` call is
   already stable.

Dropped entirely:

- `_ExecutionIdxCounter` in `context.py` — not a bugfix, an unnecessary
  semantic change.
- `ShufflingBatcher` base-seed pinning in `batcher.py` — defensive only; can
  be revisited if evidence of mid-iteration drift surfaces in practice.

A small doc-only follow-up to Ray may still be worthwhile, covering:

- Per-Dataset `_execution_idx` semantics.
- The checkpoint-resume pattern: set global before build, read
  `ds._plan._context._execution_idx` after.
- Optional: promote a public helper like `ds.execution_idx()` so users don't
  touch `_plan._context`.

## Implications for the reproducibility workspace

Applied state:

- `src/ray_repro/checkpoint_store.py` persists three pieces of state at
  the store root: per-epoch `epoch_{N}/segment_*.parquet` (consumed row
  hashes), `execution_idx.txt` (the counter), and `current_epoch.txt`
  (the epoch to resume on). The latter two are what let a crashed
  process be picked up by a fresh one that knows nothing beyond the
  store root.
- `src/ray_repro/common.py` exposes
  `run_epochs_with_resume(cfg, total_epochs, store, crash_at=...)`. It
  loads the counter from `execution_idx.txt`, loads the resume point
  from `current_epoch.txt`, and for every epoch sets the global
  `DataContext._execution_idx` before building, then wraps the dataset
  with a `row_hash` filter if that epoch already has segments on disk.
  On successful completion of an epoch it advances both counters and
  writes them back.
- `src/ray_repro/example2_checkpoint.py` accepts
  `--crash-after EPOCH:BATCHES` and `--reset`. A crash marker flushes
  the partial segment, writes `current_epoch=EPOCH` (without advancing),
  and exits 1; a resume invocation starts with no flags and follows
  what's on disk.
- `src/ray_repro/run_example2_crash_resume.py` runs the CLI twice in
  subprocesses and asserts that (a) the first exits 1 after writing the
  expected partial, (b) the second picks up at the recorded epoch, (c)
  the resumed epoch ends with ≥2 segments whose union covers every
  fixture row exactly once, and (d) the post-resume counters reflect
  every completed epoch.
- `src/ray_repro/demo_file_shuffle_rebuild.py` stays as a "don't do this
  naively" cautionary example: rebuilding `read_parquet(...)` without
  persisting `_execution_idx` gives you the same shuffle every time, by
  design.
- `src/ray_repro/demo_exec_idx_roundtrip.py` stays as the minimal
  "this is how to do it right" counterpart — the same round-trip
  Example 2 does, but without a real checkpoint store.
- `guide.md` has been updated to document the corrected mental model
  (Ray Data's reseed is per-Dataset; resume is the user's
  responsibility), the three pieces of on-disk state, and the
  crash/resume flow.

Empirical confirmation, with **no patches applied to Ray**:

```
$ python -m ray_repro.run_example2_crash_resume --seed 42 \
      --total-epochs 3 --crash-after 1:10
...
Crash at epoch 1 after 10 batches: OK
Resume picked up at epoch 1 and completed through epoch 2: OK
Resumed epoch 1 has 2 segments (crash partial + resume remainder),
    2000/2000 rows, disjoint + union OK
Final store state: current_epoch=3 execution_idx=3
All crash/resume invariants: OK
```

## Summary of the correction

The initial instinct to "fix the counter to be driver-global" was solving the
wrong problem. The counter is correctly per-Dataset; the missing piece was a
checkpoint-resume protocol around it, which is (a) the user's concern anyway
and (b) already expressible on master today with a one-line pre-build
assignment and a one-line post-execution read.
