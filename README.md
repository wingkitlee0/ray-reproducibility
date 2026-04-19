# ray-reproducibility

Examples demonstrating deterministic Ray Data ingestion, built on the
`kit/repro-example` development branch of Ray.

See [`guide.md`](./guide.md) for the design background.

## Prerequisites

An editable install of the development branch
[`wingkitlee0/ray@kit/repro-example`](https://github.com/wingkitlee0/ray/tree/kit/repro-example)
is required. It adds `include_row_hash` to `read_parquet` and
`RandomSeedConfig` support to the iterator APIs.

```bash
# In your ray checkout, on the kit/repro-example branch:
pip install -e python/
```

Then, from this repo:

```bash
pip install -e .
```

## Example 1: Reproducibility

Demonstrates that a Ray Data pipeline produces a deterministic per-epoch
row order across independent script invocations.

```bash
# Single run: prints a fingerprint per epoch.
python -m ray_repro.example1_reproducibility --seed 42 --epochs 3

# Two-run driver: runs the script twice in subprocesses and asserts that
# the per-epoch fingerprints match exactly.
python -m ray_repro.run_example1_twice --seed 42 --epochs 3
```

Expected behavior:

- Two runs with the same `--seed` produce identical per-epoch fingerprints.
- Within a single run, each epoch's fingerprint differs from the others
  (thanks to `reseed_after_execution=True`).

Ray is pinned to a single CPU (`--num-cpus 1`) so that task-completion
ordering is naturally deterministic. Parallel execution would still
produce the same *set* of rows per epoch but could reorder them across
tasks; matching fingerprints there would require
`DataContext.get_current().execution_options.preserve_order = True`.

### Continuous ordering metrics

To see *how much* parallelism perturbs ordering (rather than just "does
the fingerprint match"), sweep over `--num-cpus` and compare against the
`num_cpus=1` reference:

```bash
python -m ray_repro.compare_ordering \
    --seed 42 --epochs 3 --num-cpus 1 2 4 8 \
    --shuffle-buffer 32
```

Reported per `(num_cpus, epoch)`:

- `exact_match_fraction` -- fraction of rows at the identical position.
- `spearman_rho`         -- Spearman rank correlation (1.0 = identical,
  ~0 = uncorrelated, -1.0 = reversed).
- `displacement_score`   -- `1 - mean(|Δrank|) / E_random`, clamped to
  `[0, 1]`. Captures the intuition that extra workers usually shuffle
  rows *locally* rather than teleporting them across the epoch.

Example sweep (N=2000 rows, 8 parquet files):

```
num_cpus |   exact | spearman | disp_score
       1 |   1.000 |    1.000 |      1.000
       2 |   0.001 |    0.759 |      0.501
       4 |   0.000 |    0.492 |      0.327
       8 |   0.000 |    0.589 |      0.379
```

#### Note on `randomize_block_order()`

This stage is disabled by default (`enable_randomize_block_order=False`).
It is an all-to-all operation: it gathers every `RefBundle` from
upstream and applies a seeded shuffle. Because the *input list* to that
shuffle depends on block arrival order (which is non-deterministic under
parallelism), enabling it causes any parallelism to cascade into a
globally different permutation -- the continuous ordering score
collapses to ~0. Re-enable it with `--randomize-block-order` only when
you also plan to set
`DataContext.get_current().execution_options.preserve_order = True`.

## Example 2: Data Pipeline Checkpointing across Process Restarts

Demonstrates a realistic crash/resume cycle: one process consumes some
batches and exits, a fresh process restarts, and the rebuilt pipeline
skips every row that was already consumed while continuing the reseed
sequence from where it left off. Filtering is by `row_hash` (from
`include_row_hash=True`), so it composes cleanly with both
`FileShuffleConfig` and the local shuffle buffer.

```bash
# Clean single run: 3 epochs, fresh store.
python -m ray_repro.example2_checkpoint --seed 42 --total-epochs 3 --reset

# Simulated crash: finish epoch 0, consume 10 batches of epoch 1, exit 1.
python -m ray_repro.example2_checkpoint --seed 42 --total-epochs 3 \
    --reset --crash-after 1:10

# Resume the above: no --reset, no --crash-after. The store's
# current_epoch=1 and 640 pre-consumed row_hashes under epoch_1/ drive
# the filter + drain logic automatically.
python -m ray_repro.example2_checkpoint --seed 42 --total-epochs 3

# End-to-end driver: runs the crash and resume subprocesses and asserts
# every invariant (partial segment written, resume picks up at the right
# epoch, per-epoch row sets are disjoint and union-complete, final
# counters match the total number of completed epochs).
python -m ray_repro.run_example2_crash_resume --seed 42 --total-epochs 3 \
    --crash-after 1:10
```

Guarantees per epoch:

- `union(segments) == expected_set` — no row dropped, even across a crash.
- Segments are pairwise disjoint — no row consumed twice.

`DataContext._execution_idx` is per-Dataset in Ray by design, so the
example persists the counter (and the current epoch index) alongside the
seen-hash segments and restores them before each rebuild. See
[`docs/execution-idx-semantics.md`](docs/execution-idx-semantics.md) for
the reasoning and `src/ray_repro/demo_exec_idx_roundtrip.py` for a
minimal, checkpoint-store-free version of the same round-trip.
