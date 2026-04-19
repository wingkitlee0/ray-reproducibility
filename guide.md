# Ray Data and Train: Reproducibility and Data Checkpointing Examples

This guide describes two related examples to build with Ray Data and Ray Train. Each example is
self-contained and can be generated independently by an AI agent. They share a common background
on Ray Data randomness controls (see [Shared Background](#shared-background)).

- [Example 1: Reproducibility](#example-1-reproducibility)
- [Example 2: Data Pipeline Checkpointing](#example-2-data-pipeline-checkpointing)

---

## Shared Background

Both examples rely on recent Ray Data features for controlling randomness deterministically.

### `RandomSeedConfig`

Ray recently introduced [`RandomSeedConfig`](https://docs.ray.io/en/latest/data/api/doc/ray.data.RandomSeedConfig.html#ray.data.RandomSeedConfig),
which allows the RNG to reseed deterministically for repeated execution. Users provide an initial
base seed and can expect pseudo-randomness across executions.

A common use case: when training a deep learning model, each epoch triggers a new Ray Data
pipeline execution. With `reseed_after_execution=True`, the user gets a new (but deterministic)
seed for each pipeline execution.

### A typical Ray Data / Ray Train pipeline

1. `ray.data.read_parquet` — use `FileShuffleConfig` with a base seed and
   `reseed_after_execution=True` to control file shuffling.
2. Optional `randomize_block_order()` etc., which support `RandomSeedConfig`.
3. In a Ray Train use case, a single Ray dataset is split into multiple iterators via
   `streaming_split`. Each Trainer process consumes batches from its iterator synchronously.
4. Before feeding batches into the model, users can also specify a local shuffle via
   `local_shuffle_buffer`.

### Development branch used in these examples

The local shuffle buffer does not yet support `RandomSeedConfig` on `master`. Both examples use
the branch [`wingkitlee0/ray@kit/repro-example`](https://github.com/wingkitlee0/ray/tree/kit/repro-example),
which adds:

1. A row-hash column via `include_row_hash` when reading parquet files.
2. `RandomSeedConfig` support in `iter_torch_batches()` and other data iterator APIs.

---

## Example 1: Reproducibility

### Goal

Demonstrate that a Ray Data pipeline produces a **deterministic data ingestion order** across
runs when configured with the features above. Given the same base seed, two independent
invocations of the script should yield the identical sequence of rows/batches across all epochs.

This example is **pipeline-only** — no real model, no Ray Train, no PyTorch model code required.
A plain iteration loop (e.g., `iter_batches()` or `iter_torch_batches()`) is sufficient.

### Key ideas

1. Configure every randomness source in the pipeline with a seeded `RandomSeedConfig` (or
   equivalent), including:
   - `FileShuffleConfig` in `read_parquet`
   - `randomize_block_order()` (if used)
   - `iter_batches(..., local_shuffle_buffer_size=..., random_seed_config=...)` (or
     `iter_torch_batches` with the same parameters)
2. Enable `reseed_after_execution=True` so that each epoch gets a new but deterministic seed,
   ensuring row ordering varies across epochs but is reproducible across runs.
3. Use `include_row_hash=True` in `read_parquet` so each row carries a stable hash the demo can
   use as an order-dependent fingerprint.

### What the example should show

The demo script iterates the pipeline for N epochs and prints/records an **order-dependent
fingerprint** per epoch, such as:

- A running hash of the row-hash column across batches (e.g., feed each batch's row hashes into
  a rolling SHA-256 / xxhash).
- Or simply the first K row hashes of each epoch.

Expected behavior:

- Running the script **twice with the same seed** produces identical per-epoch fingerprints.
- Fingerprints **differ across epochs** within a single run (thanks to `reseed_after_execution`).
- A final `assert` or diff between two run logs confirms reproducibility.

No torch model, optimizer, or training loop is needed — just `ray.data` + an iterator.

---

## Example 2: Data Pipeline Checkpointing

### Goal

Build on the reproducibility setup to demonstrate **data pipeline checkpointing**: when a run
resumes from a checkpoint, the pipeline skips rows already seen **within the current epoch** so
they are not re-consumed. Tracking is **per-epoch** — once an epoch completes, the seen-row
state resets for the next epoch.

This example is also **pipeline-only** — no torch model, no Ray Train. A plain iteration loop
over `iter_batches()` (optionally split by `streaming_split` to simulate multiple consumers) is
sufficient to demonstrate the checkpoint-and-resume flow.

### Why `filter()` instead of offset skipping

Offset-based resume ("skip the first N rows") is not reliable here because:

- `FileShuffleConfig` reorders files per epoch execution.
- `local_shuffle_buffer` further perturbs within-epoch ordering.

So an "offset" has no stable meaning across a fresh pipeline execution. A content-based
`filter()` on `row_hash` is invariant to both shuffles and composes cleanly with the rest of
the pipeline, making it the right primitive for a quick demo.

### Key ideas

1. At each epoch, the Ray Data pipeline is reloaded and executed. Ensure:
   1. Row ordering differs across epochs (via `reseed_after_execution=True`).
   2. On resume from a mid-epoch checkpoint, the pipeline skips rows already seen **in that
      epoch** so only unseen rows are iterated.
   3. When the epoch finishes, seen-row tracking is cleared before starting the next epoch.

2. A simple approach to per-epoch data checkpointing:
   1. **Track seen rows for the current epoch** — as the demo iterates batches, record the row
      hashes (from `include_row_hash`) of consumed rows. If using `streaming_split` with
      multiple iterators, each consumer writes its own per-consumer file in parallel. Scope the
      files to the current epoch (e.g., `epoch=<N>/consumer=<K>.parquet`).
   2. **Rebuild a global filter** from the current epoch's per-consumer files on resume.
      Optimize as needed (e.g., sort, partition, bloom filter).
   3. **Apply the global filter** as a `filter()` step in the pipeline so already-seen rows are
      excluded from the current epoch.
   4. **Reset between epochs** — at the start of each new epoch, drop / ignore prior epochs'
      seen-row state so the full dataset is eligible again (with fresh shuffling).

### What the example should show

- The demo iterates the pipeline and "checkpoints" mid-epoch by stopping after M batches and
  persisting the seen row hashes for that epoch.
- On resume, the pipeline is reconstructed with the same seed config plus a `filter()` over the
  current epoch's seen hashes.
- The union of (pre-checkpoint rows) ∪ (post-resume rows) equals the full set of rows for that
  epoch — no duplicates, no omissions. A simple assertion over row-hash sets confirms this.
- When the epoch completes, seen-row tracking is reset; the next epoch starts with a clean slate
  and (thanks to reseeding) a different row order.

No torch model, optimizer, or Ray Train loop is needed — just `ray.data` + iterators + a small
filesystem-backed seen-hash store.

---

## Example 1 — Implementation Notes

This section summarizes what has been implemented for Example 1 in this repo,
the design choices behind it, and the empirical behavior observed when
sweeping `--num-cpus`. See *Example 2 — Implementation Notes* below for the
checkpoint/resume example and *`docs/execution-idx-semantics.md`* for the
per-Dataset vs. driver-global discussion that shaped Example 2.

### Repository layout

```
src/ray_repro/
    fixture.py                      # generate tiny parquet dataset (8 files × 250 rows)
    common.py                       # PipelineConfig, build_dataset, iter_epoch_batches,
                                    # fingerprint_epoch, collect_row_hashes, ordering_metrics
    example1_reproducibility.py     # single-script demo: N epochs → N fingerprints
    run_example1_twice.py           # driver: runs example1 twice in subprocesses,
                                    # asserts per-epoch fingerprints match exactly
    compare_ordering.py             # driver: sweeps --num-cpus and reports continuous
                                    # ordering metrics vs. num_cpus=1 reference
```

### The pipeline and how each stage behaves under parallelism

The default pipeline (`common.PipelineConfig`) is:

```python
ds = ray.data.read_parquet(
    data_dir,
    shuffle=FileShuffleConfig(seed=S, reseed_after_execution=True),
    include_row_hash=True,
)
# optional:
# ds = ds.randomize_block_order(seed=RandomSeedConfig(...))
# iteration:
ds.iter_batches(
    batch_size=...,
    local_shuffle_buffer_size=...,
    local_shuffle_seed=RandomSeedConfig(seed=S, reseed_after_execution=True),
)
```

Three stages, three behaviors under parallel execution:

| Stage | Behavior under `num_cpus > 1` |
| --- | --- |
| `FileShuffleConfig` in `read_parquet` | Deterministic. Picks which file each task reads based only on the seed and execution index; no dependence on task timing. |
| `randomize_block_order()` | **Non-deterministic under parallelism.** All-to-all: gathers every `RefBundle` upstream and applies `rng.shuffle(bundles)`. The *input list order* depends on block arrival order, so any task scheduling jitter produces a globally different permutation. Disabled by default. |
| `local_shuffle_buffer_size` + `local_shuffle_seed` | Locally non-deterministic under parallelism. A seeded reservoir-style shuffle whose output depends on the arrival order of upstream batches into the buffer. Perturbation stays bounded by the buffer size. |

Consequence: to keep reproducibility across arbitrary `num_cpus` in the full
pipeline, either run with 1 CPU or set
`DataContext.get_current().execution_options.preserve_order = True`.

### Minimal pipeline: `read_parquet(shuffle=FileShuffleConfig(...))`

The minimal pipeline strips both downstream shuffles and uses only file-level
shuffling:

```python
ds = ray.data.read_parquet(
    data_dir,
    shuffle=FileShuffleConfig(seed=42, reseed_after_execution=True),
    include_row_hash=True,
)
for batch in ds.iter_batches(batch_size=64, batch_format="pyarrow"):
    consume(batch["row_hash"])
```

In `compare_ordering` terms this is:

```bash
python -m ray_repro.compare_ordering \
    --seed 42 --epochs 3 --num-cpus 1 2 4 8 \
    --shuffle-buffer 0 \
    --file-shuffle --no-randomize-block-order
```

Observed metrics on the 2000-row fixture:

```
num_cpus |   exact | spearman | disp_score
       1 |   1.000 |    1.000 |      1.000
       2 |   1.000 |    1.000 |      1.000
       4 |   0.750 |    0.812 |      0.750
       8 |   0.750 |    0.812 |      0.750
```

Interpretation:

- Most epochs are *bit-identical* to the `num_cpus=1` reference, even at 4–8
  CPUs. With only 8 files and tasks roughly in submission order, the Ray
  streaming executor usually delivers blocks in file-task order.
- When ordering does diverge (epoch 2 at `num_cpus=4`; epoch 0 at
  `num_cpus=8`), the disturbance is exactly one adjacent-block swap: two
  250-row blocks trade places, giving `mean|Δrank| = 500` and
  `displacement_score = 0.25`. No row ever teleports far from its reference
  position.
- The pair-swap structure confirms the intuition that extra CPUs introduce
  *local* ordering noise rather than globally randomizing the epoch.

Note that `exact_match_fraction` is a harsh metric here: a single adjacent
swap drops it from 1.000 to 0.250, even though every row is within one
block of its reference position. `spearman_rho` and `displacement_score`
degrade more gracefully and are the better metrics when the goal is
"how close is the ordering to the reference", not "is it bit-identical".

Adding `local_shuffle_buffer_size=32` to the same sweep produces the smooth
monotonic-ish degradation expected when local shuffling is active:

```
num_cpus |   exact | spearman | disp_score
       1 |   1.000 |    1.000 |      1.000
       2 |   0.001 |    0.759 |      0.501
       4 |   0.000 |    0.493 |      0.327
       8 |   0.000 |    0.621 |      0.390
```

`exact` collapses to ~0 because the buffer reshuffles every row slightly;
the continuous metrics still show rows stay well within ~1/3 of the random
baseline displacement.

### Continuous ordering metrics

Defined in `common.ordering_metrics(observed, reference)` where both inputs
are 1-D uint64 arrays (per-epoch `row_hash` sequences from
`common.collect_row_hashes`):

- `exact_match_fraction`: fraction of rows at the same index as the
  reference. 1.0 = identical. Sensitive to any perturbation, so useful
  mainly as a ceiling check.
- `spearman_rho`: Spearman rank correlation between observed and reference
  positions. 1.0 = identical, ~0 = uncorrelated, −1 = reversed.
- `displacement_score`: `1 − mean(|Δrank|) / E_random[|Δrank|]` clamped to
  `[0, 1]`, where `E_random = (n² − 1) / (3n) ≈ n/3` is the expected mean
  rank displacement of a uniformly random permutation. 1.0 = identical;
  0.0 means the average row is as far from its reference position as it
  would be under a random shuffle. Best single metric for "rows moved
  locally, not globally".
- `mean_rank_displacement`: raw mean `|observed_pos − reference_pos|` in
  rank units; useful as a sanity-check of the normalized scores.

### Scripts and typical invocations

Single run (prints one fingerprint per epoch):

```bash
python -m ray_repro.example1_reproducibility --seed 42 --epochs 3
```

Cross-run reproducibility check (runs twice in fresh subprocesses, asserts
per-epoch fingerprints match):

```bash
python -m ray_repro.run_example1_twice --seed 42 --epochs 3
```

Continuous ordering sweep:

```bash
python -m ray_repro.compare_ordering \
    --seed 42 --epochs 3 --num-cpus 1 2 4 8 --shuffle-buffer 32
```

### Conventions

- `--num-cpus 1` is the default so task-completion order is deterministic
  without needing `preserve_order`. Users interested in parallel
  reproducibility can set it via `DataContext`.
- All boolean flags use `argparse.BooleanOptionalAction`
  (`--file-shuffle` / `--no-file-shuffle`, etc.), so attribute names are
  positive (`args.file_shuffle`) and defaults are explicit.
- Per-epoch `row_hash` sequences can be dumped via `--sequences-dir` on
  `example1_reproducibility`; the sweep driver uses this to avoid
  re-running the pipeline just to compute metrics.

## Example 2 — Implementation Notes

### What the example demonstrates

A realistic crash/resume story: one process consumes part of an epoch and
dies, a *fresh* process restarts and continues from exactly where the
first left off, skipping rows already consumed.

Concretely, with ``--crash-after 1:10 --total-epochs 3``:

1. Process A (fresh store): runs epoch 0 to completion, then consumes 10
   batches (= 640 rows) of epoch 1 and exits with code 1.
2. On-disk state after process A: 1 completed epoch's hashes, 1 partial
   segment under ``epoch_1/``, ``current_epoch=1``, ``execution_idx=1``.
3. Process B (no reset): reads the state, restores
   ``DataContext._execution_idx = 1``, rebuilds the pipeline for epoch 1,
   wraps it with a ``row_hash`` filter that drops the 640 already-seen
   rows, drains the remaining 1360, then runs epoch 2 normally.
4. Final state: every epoch's segment set covers the full 2000 rows
   exactly once (disjoint + union-complete). ``current_epoch=3``,
   ``execution_idx=3``.

This matches the shape of training checkpointing: mid-epoch failures are
resumed without either double-ingesting or dropping any rows.

### Added pieces

- `src/ray_repro/checkpoint_store.py` — `SeenHashStore`: parquet segments
  under `epoch_{N}/` for consumed row hashes, plus top-level
  `execution_idx.txt` and `current_epoch.txt` so a restarting process
  knows which epoch to resume on and which seed to restore.
- `src/ray_repro/common.py`:
  - `filter_by_row_hash(ds, excluded)` drops rows whose `row_hash` is in
    `excluded`. Implemented as `map_batches(...)` using `pc.is_in` at the
    `pyarrow.Table` level; a `ds.filter(lambda row: ...)` path was tried
    first but Ray Data's per-row predicate route serialized the hash into
    a pickled bytes column, so batch-level Arrow filtering was required.
  - `enumerate_row_hashes(data_dir)` returns every `row_hash` in canonical
    read order (no shuffle). Used as ground truth.
  - `run_epochs_with_resume(cfg, total_epochs, store, crash_at=...)` is
    the resume-aware orchestrator: it reads `current_epoch` /
    `execution_idx` from the store, runs each epoch by either starting
    fresh (no segments yet) or continuing (existing segments filtered
    out), optionally raises `CrashInjected` at the supplied
    `(epoch, batches)` marker, and advances the counters only on
    *completed* epochs.
- `src/ray_repro/example2_checkpoint.py` — the CLI. Flags:
  `--total-epochs`, `--crash-after EPOCH:BATCHES`, `--reset`,
  `--store-root`, plus the usual `--seed` / `--batch-size` /
  `--shuffle-buffer` / `--file-shuffle` / `--randomize-block-order` /
  `--num-cpus` / `--json`. Returns exit code 1 when a crash is injected
  (so an outer driver can detect the simulated failure), 0 on clean
  completion.
- `src/ray_repro/run_example2_crash_resume.py` — driver that runs the CLI
  twice against a shared store (first with `--reset --crash-after ...`,
  then with no reset and no crash) and asserts the full crash/resume
  invariants.
- `src/ray_repro/demo_file_shuffle_rebuild.py` and
  `src/ray_repro/demo_exec_idx_roundtrip.py` — minimal scripts that
  demonstrate, respectively, why naive `read_parquet(...)` rebuilds all
  hit the same shuffle, and how a three-line pre-build / post-execute
  round-trip fixes it.

### Per-epoch invariants

For every epoch `N` that completes (whether in one shot or across a
crash/resume split), the orchestrator verifies:

- `union of epoch_{N}/segment_*.parquet == expected_set` — no row dropped.
- The segment sets are pairwise disjoint — no row consumed twice.
- `rows_consumed == 2000` (= full fixture size).

In the clean case an epoch has 1 segment and the above is trivial. In the
crash-resume case the interrupted epoch has ≥2 segments (one per process
that touched it) and the same invariants hold non-trivially.

### State pieces and their scopes

| State | Location | Scope | Reset when |
| --- | --- | --- | --- |
| Seen row hashes | `epoch_{N}/segment_*.parquet` | Per-epoch | `store.clear_epoch(N)` at the start of epoch `N` if you want fresh behavior; retained on a crash |
| `_execution_idx` counter | `execution_idx.txt` | Per-run | Only `store.clear_all()` (invoked on `--reset`) |
| Current epoch | `current_epoch.txt` | Per-run | Only `store.clear_all()` |

The counter advances exactly once per *completed* epoch, so the reseed
sequence is identical whether epoch `N` finished in one process or was
split across two. The "current epoch" pointer only advances on successful
completion, which is why a crashed run leaves it on the interrupted
epoch and a resume knows to pick up there.

### Why this needs user-managed checkpoint state (and not a Ray-side fix)

`DataContext._execution_idx` is per-Dataset by design: a rebuilt
`read_parquet(...)` is semantically a new pipeline and legitimately starts
the counter at 0. Continuing the seed sequence across rebuilds is a
checkpoint-resume concern, and checkpoint-resume state is the user's job.
See [`docs/execution-idx-semantics.md`](docs/execution-idx-semantics.md)
for the full discussion, including an earlier iteration of this example
that tried to "fix" it with a driver-wide shared counter in Ray core and
why that was reverted.

The upshot: the only Ray-side changes this example depends on are the
two already-landing PRs on `kit/repro-example` — `include_row_hash` and
`iter_batches` local-shuffle reproducibility. There is no Ray-core
bugfix piggybacking on Example 2.

### Scripts and typical invocations

Single run (clean, fresh store):

```bash
python -m ray_repro.example2_checkpoint --seed 42 --total-epochs 3 --reset
```

Single run with injected crash:

```bash
# Exits 1 after consuming 10 batches of epoch 1. Store is left in a
# partially-consumed state.
python -m ray_repro.example2_checkpoint --seed 42 --total-epochs 3 \
    --reset --crash-after 1:10

# Second invocation: no --reset, no --crash-after. Reads the stored
# state, filters the 640 already-consumed rows out of epoch 1's rebuild,
# drains the remainder, then runs epoch 2 normally.
python -m ray_repro.example2_checkpoint --seed 42 --total-epochs 3
```

End-to-end crash/resume driver (runs both above in subprocesses and
asserts all invariants):

```bash
python -m ray_repro.run_example2_crash_resume --seed 42 --total-epochs 3 \
    --crash-after 1:10
```

Observed output on the 2000-row fixture:

```
[run 1 exits with code 1 after writing the partial segment]
[run 2 resumes]

epoch=1 segments=2 rows=2000/2000 disjoint=True union_ok=True fp=3ef65fe3c7a6f1c3:2000
epoch=2 segments=1 rows=2000/2000 disjoint=True union_ok=True fp=e6d0624a69a0be74:2000

Crash at epoch 1 after 10 batches: OK
Resume picked up at epoch 1 and completed through epoch 2: OK
Resumed epoch 1 has 2 segments (crash partial + resume remainder),
    2000/2000 rows, disjoint + union OK
Final store state: current_epoch=3 execution_idx=3
All crash/resume invariants: OK
```

A nice cross-example consistency check: the epoch-2 fingerprint
(`e6d0624a69a0be74:2000`) is identical to what a no-crash run produces
and to what Example 1 would yield for its equivalent epoch under the
same config. The damage of a mid-epoch crash is strictly bounded to the
epoch that was interrupted; every subsequent epoch is bit-for-bit what
an uninterrupted run would have produced.
