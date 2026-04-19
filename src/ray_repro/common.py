"""Shared pipeline + fingerprint helpers for the reproducibility examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import ray
import xxhash
from ray.data import DataContext, Dataset, FileShuffleConfig, RandomSeedConfig


@dataclass(frozen=True)
class PipelineConfig:
    """User-facing knobs for the deterministic pipeline.

    All three shuffle stages are driven from the same base ``seed``. With
    ``reseed_after_execution=True``, each epoch (i.e. each full pipeline
    execution) derives a fresh-but-deterministic per-epoch seed, so row
    ordering varies across epochs within a run while remaining identical
    across independent runs with the same base seed.
    """

    data_dir: Path
    seed: int = 42
    batch_size: int = 64
    local_shuffle_buffer_size: int = 256
    reseed_after_execution: bool = True
    # Toggles -- useful for isolating how each stage reacts to parallelism.
    enable_file_shuffle: bool = True
    # randomize_block_order is an all-to-all op whose seeded permutation is
    # applied over a parallelism-dependent arrival order, so it collapses any
    # continuous ordering metric to ~0 the moment num_cpus > 1. It's left off
    # by default; enable it explicitly if you also set
    # ``DataContext.get_current().execution_options.preserve_order = True``.
    enable_randomize_block_order: bool = False

    def seed_config(self) -> RandomSeedConfig:
        return RandomSeedConfig(
            seed=self.seed,
            reseed_after_execution=self.reseed_after_execution,
        )

    def file_shuffle(self) -> FileShuffleConfig:
        return FileShuffleConfig(
            seed=self.seed,
            reseed_after_execution=self.reseed_after_execution,
        )


def build_dataset(cfg: PipelineConfig) -> Dataset:
    """Build the deterministic Ray Data pipeline used by the examples.

    Every randomness source is seeded from the same base via
    ``RandomSeedConfig`` / ``FileShuffleConfig``. ``include_row_hash=True``
    attaches a stable ``row_hash`` column we use as the order-dependent
    fingerprint.

    When ``cfg.enable_randomize_block_order`` is ``False``, the
    all-to-all stage is skipped; this lets the ``compare_ordering`` driver
    reveal a smooth degradation of ordering with parallelism, because the
    remaining perturbation is local (task interleaving + local shuffle
    buffer) rather than a globally seeded permutation over a
    parallelism-dependent block arrival order.
    """
    ds = ray.data.read_parquet(
        str(cfg.data_dir),
        shuffle=cfg.file_shuffle() if cfg.enable_file_shuffle else None,
        include_row_hash=True,
    )
    if cfg.enable_randomize_block_order:
        ds = ds.randomize_block_order(seed=cfg.seed_config())
    return ds


def iter_epoch_batches(ds: Dataset, cfg: PipelineConfig):
    """Iterate one epoch's batches with seeded local shuffle.

    A local_shuffle_buffer_size of 0 (or None) disables the local shuffle
    entirely; we normalize both to ``None`` for Ray.
    """
    buffer_size = cfg.local_shuffle_buffer_size or None
    return ds.iter_batches(
        batch_size=cfg.batch_size,
        batch_format="pyarrow",
        local_shuffle_buffer_size=buffer_size,
        local_shuffle_seed=cfg.seed_config() if buffer_size else None,
    )


def _format_fingerprint(digest_hex: str, row_count: int) -> str:
    """Stable on-disk / log-line format for an epoch fingerprint.

    Embedding ``row_count`` guards against silent row drops / duplications:
    a matching digest with a mismatched count still fails comparison.
    """
    return f"{digest_hex}:{row_count}"


def fingerprint_sequence(seq: np.ndarray) -> str:
    """Compute the epoch fingerprint from a materialized ``row_hash`` sequence.

    ``seq`` must be a 1-D array coercible to ``uint64``. Permuting rows
    changes the digest; the batch framing is irrelevant because only the
    final byte sequence is hashed.
    """
    arr = np.ascontiguousarray(seq, dtype=np.uint64)
    h = xxhash.xxh64()
    h.update(arr.tobytes(order="C"))
    return _format_fingerprint(h.hexdigest(), int(arr.shape[0]))


def fingerprint_epoch(batches: Iterable) -> str:
    """Streaming fingerprint over an epoch's batches.

    Equivalent to ``fingerprint_sequence(collect_row_hashes(batches))`` but
    without materializing the full sequence in memory -- useful for large
    datasets where we only need the digest, not the raw hashes.
    """
    h = xxhash.xxh64()
    row_count = 0
    for batch in batches:
        # batch["row_hash"] is a pyarrow ChunkedArray / Array; materialize to
        # a contiguous little-endian uint64 buffer for a stable byte feed.
        arr = np.asarray(batch["row_hash"], dtype=np.uint64)
        h.update(arr.tobytes(order="C"))
        row_count += arr.shape[0]
    return _format_fingerprint(h.hexdigest(), row_count)


def collect_row_hashes(
    batches: Iterable,
    *,
    max_batches: int | None = None,
) -> np.ndarray:
    """Materialize the ``row_hash`` sequence as a 1-D uint64 array.

    Order matches the order rows are yielded by the iterator -- i.e., the
    exact ingestion order a training loop would see. If ``max_batches`` is
    set, iteration stops after that many batches (used by Example 2 to
    simulate a mid-epoch checkpoint). Unlike :func:`fingerprint_epoch`, this
    preserves enough information to compute ordering metrics or feed back
    into a content-based filter.
    """
    chunks: list[np.ndarray] = []
    for i, batch in enumerate(batches):
        if max_batches is not None and i >= max_batches:
            break
        chunks.append(np.asarray(batch["row_hash"], dtype=np.uint64))
    if not chunks:
        return np.empty(0, dtype=np.uint64)
    return np.concatenate(chunks)


def filter_by_row_hash(
    ds: Dataset,
    excluded: Iterable[int] | np.ndarray,
) -> Dataset:
    """Return a dataset that drops rows whose ``row_hash`` is in ``excluded``.

    Used by Example 2 to resume a partially-consumed epoch: the previously
    seen row_hashes are passed here so the re-executed pipeline skips them.
    Filtering happens at the ``pyarrow.Table`` level (via ``map_batches``
    with ``pc.is_in``) so the ``uint64`` schema is preserved; a per-row
    Python predicate would round-trip through object dicts and corrupt the
    column. A production deployment would prefer a bloom filter or a
    pushdown expression for larger exclusion sets.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(excluded, np.ndarray):
        arr = np.asarray(excluded, dtype=np.uint64)
    else:
        arr = np.fromiter((int(h) for h in excluded), dtype=np.uint64)
    if arr.size == 0:
        return ds
    excluded_pa = pa.array(arr, type=pa.uint64())

    def _drop_seen(batch: "pa.Table") -> "pa.Table":
        mask = pc.invert(pc.is_in(batch.column("row_hash"), value_set=excluded_pa))
        return batch.filter(mask)

    return ds.map_batches(_drop_seen, batch_format="pyarrow")


def enumerate_row_hashes(data_dir: Path) -> np.ndarray:
    """Return every ``row_hash`` the dataset contains, in canonical read order.

    No shuffling is applied, so the result only depends on the fixture files
    -- not on the current ``DataContext._execution_idx``. Example 2 uses this
    as ground-truth when asserting that the union of pre-checkpoint and
    post-resume rows equals the full epoch.
    """
    ds = ray.data.read_parquet(str(data_dir), include_row_hash=True)
    return collect_row_hashes(ds.iter_batches(batch_format="pyarrow"))


def run_epochs(
    cfg: PipelineConfig,
    epochs: int,
    *,
    sequences_dir: Path | None = None,
) -> list[str]:
    """Run ``epochs`` full passes over the pipeline and return per-epoch fingerprints.

    Each epoch's ``row_hash`` sequence is materialized once and used both to
    compute the fingerprint (via :func:`fingerprint_sequence`) and -- when
    ``sequences_dir`` is given -- to persist ``epoch_{i}.npy`` for downstream
    metric computation. This keeps the demo scripts free of hashing or
    serialization logic.
    """
    ds = build_dataset(cfg)
    if sequences_dir is not None:
        sequences_dir.mkdir(parents=True, exist_ok=True)

    fingerprints: list[str] = []
    for epoch in range(epochs):
        seq = collect_row_hashes(iter_epoch_batches(ds, cfg))
        if sequences_dir is not None:
            np.save(sequences_dir / f"epoch_{epoch}.npy", seq)
        fingerprints.append(fingerprint_sequence(seq))
    return fingerprints


@dataclass(frozen=True)
class EpochResult:
    """Outcome of one epoch after it has been fully consumed (possibly across
    multiple processes via crash/resume).

    ``segments`` is the number of on-disk segments that together cover the
    epoch's row set -- ``1`` for an epoch that ran to completion in a single
    process, ``2+`` for epochs that were interrupted and resumed (one
    segment per crash).

    Invariants (checked by :func:`run_epochs_with_resume`):

    - ``rows_consumed == expected_rows``: the epoch covered every fixture
      row exactly once.
    - ``segments_disjoint``: no row appears in more than one segment, i.e.
      phase-N always saw rows phase-(<N) did not.
    - ``union_equals_expected``: union across segments equals the fixture.
    """

    epoch: int
    segments: int
    rows_consumed: int
    expected_rows: int
    segments_disjoint: bool
    union_equals_expected: bool
    concat_fingerprint: str


class CrashInjected(Exception):
    """Raised internally to simulate a process crash mid-epoch.

    Caught only by the CLI entry point, which translates it to a non-zero
    exit code so an outer driver can observe the "crash" as a real
    subprocess failure without Python printing a traceback for expected
    demo behavior.
    """

    def __init__(self, epoch: int, batches_consumed: int):
        super().__init__(
            f"simulated crash after {batches_consumed} batches of epoch {epoch}"
        )
        self.epoch = epoch
        self.batches_consumed = batches_consumed


def _set_execution_idx(idx: int) -> None:
    """Set the per-process ``DataContext._execution_idx``.

    Every subsequently constructed ``Dataset`` will copy this value into its
    private context and use it to derive reseeded shuffle keys. This is the
    hook the checkpoint-resume contract hangs off (see
    ``docs/execution-idx-semantics.md``).
    """
    DataContext.get_current()._execution_idx = int(idx)


def _read_execution_idx(ds: Dataset) -> int:
    """Read ``_execution_idx`` from a Dataset's private plan context.

    This peeks past the public API intentionally: after execution, the
    counter lives on ``ds._plan._context`` (where Ray's callbacks bumped
    it), not on the global ``DataContext``. There is no public getter yet,
    but the shape of the field has been stable.
    """
    return int(ds._plan._context._execution_idx)


def _consume_epoch(
    cfg: PipelineConfig,
    epoch: int,
    store,
    *,
    execution_idx: int,
    crash_after_batches: int | None,
) -> np.ndarray:
    """Run one epoch from its current on-disk state to (ideally) completion.

    If segments already exist under ``epoch`` in the store, we are resuming:
    build the dataset, wrap it with :func:`filter_by_row_hash` so those rows
    are dropped, and iterate what remains. Otherwise iterate the full
    dataset.

    If ``crash_after_batches`` is reached before the epoch completes, the
    batches consumed so far are appended as a new segment, the current
    epoch is stamped to the store (so a resuming process knows not to
    advance past it), and :class:`CrashInjected` is raised. The caller is
    expected to exit non-zero at that point.

    On successful completion the new batch's hashes are appended as the
    final segment for this epoch and returned.
    """
    _set_execution_idx(execution_idx)
    seen = store.load_epoch(epoch)

    ds = build_dataset(cfg)
    if seen.size > 0:
        ds = filter_by_row_hash(ds, seen)

    chunks: list[np.ndarray] = []
    for i, batch in enumerate(iter_epoch_batches(ds, cfg)):
        chunks.append(np.asarray(batch["row_hash"], dtype=np.uint64))
        if crash_after_batches is not None and (i + 1) >= crash_after_batches:
            partial = (
                np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
            )
            store.append(epoch, partial)
            store.save_current_epoch(epoch)
            raise CrashInjected(epoch=epoch, batches_consumed=i + 1)

    final_segment = (
        np.concatenate(chunks) if chunks else np.empty(0, dtype=np.uint64)
    )
    store.append(epoch, final_segment)
    return final_segment


def run_epochs_with_resume(
    cfg: PipelineConfig,
    total_epochs: int,
    store,
    *,
    crash_at: tuple[int, int] | None = None,
) -> list[EpochResult]:
    """Drive the checkpoint/resume pipeline to ``total_epochs`` epochs.

    The on-disk state in ``store`` is the source of truth for where work
    left off:

    - ``current_epoch``: the epoch to resume on. Zero on a fresh store.
    - ``execution_idx``: the seed-advancing counter that is restored into
      ``DataContext`` before every rebuild so reseeded shuffles continue
      the sequence across rebuilds. Only advanced once per *completed*
      epoch.
    - ``epoch_{N}/segment_*.parquet``: row hashes already consumed for
      epoch ``N``. A non-empty set means "we crashed partway through this
      epoch; filter these rows out and drain the rest".

    ``crash_at`` simulates a mid-epoch crash at ``(epoch, batches_consumed)``
    for demo purposes. When the marker is reached, partial state is
    flushed and :class:`CrashInjected` is raised; outer callers that want
    to simulate a real process exit should let it propagate.

    Returns one :class:`EpochResult` per epoch completed in *this* call
    (i.e. excluding epochs that were already finished in a prior run).
    """
    expected = enumerate_row_hashes(cfg.data_dir)
    expected_set = set(int(x) for x in expected)

    current_epoch = store.load_current_epoch()
    execution_idx = store.load_execution_idx()
    results: list[EpochResult] = []

    for epoch in range(current_epoch, total_epochs):
        per_epoch_crash: int | None = None
        if crash_at is not None and crash_at[0] == epoch:
            per_epoch_crash = crash_at[1]

        _consume_epoch(
            cfg,
            epoch,
            store,
            execution_idx=execution_idx,
            crash_after_batches=per_epoch_crash,
        )

        all_hashes = store.load_epoch(epoch)
        segment_sets = _segment_sets(store, epoch)
        union = set().union(*segment_sets) if segment_sets else set()
        segments_disjoint = sum(len(s) for s in segment_sets) == len(union)

        results.append(
            EpochResult(
                epoch=epoch,
                segments=len(segment_sets),
                rows_consumed=len(union),
                expected_rows=len(expected_set),
                segments_disjoint=segments_disjoint,
                union_equals_expected=union == expected_set,
                concat_fingerprint=fingerprint_sequence(all_hashes),
            )
        )

        execution_idx += 1
        store.save_execution_idx(execution_idx)
        store.save_current_epoch(epoch + 1)

    return results


def _segment_sets(store, epoch: int) -> list[set[int]]:
    """Return one ``set[int]`` per on-disk segment for ``epoch``.

    Kept separate so tests / metrics can distinguish between "one full
    segment" (clean run) and "phase-1 + phase-2 segments" (crash and
    resume).
    """
    import pyarrow.parquet as pq

    d = store.epoch_dir(epoch)
    if not d.exists():
        return []
    files = sorted(d.glob("segment_*.parquet"))
    return [
        set(
            int(x)
            for x in np.asarray(
                pq.read_table(f, columns=["row_hash"])["row_hash"], dtype=np.uint64
            )
        )
        for f in files
    ]


def _reference_positions(observed: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """For each element of ``observed``, return its index in ``reference``.

    Requires the two arrays to contain the same set of distinct values.
    Implemented with a sorted-reference binary search so it is O(N log N)
    and works on uint64 keys without building a Python dict.
    """
    if observed.shape != reference.shape:
        raise ValueError(
            f"observed ({observed.shape}) and reference ({reference.shape}) "
            "must have the same shape"
        )
    if observed.size == 0:
        return np.empty(0, dtype=np.int64)

    ref_sort_idx = np.argsort(reference, kind="stable")
    sorted_ref = reference[ref_sort_idx]
    locations = np.searchsorted(sorted_ref, observed)
    out_of_range = locations >= sorted_ref.size
    if out_of_range.any() or not np.array_equal(
        sorted_ref[np.clip(locations, 0, sorted_ref.size - 1)], observed
    ):
        raise ValueError(
            "observed contains values not present in reference (or duplicates "
            "differ between the two). Ordering metrics require both sequences "
            "to be permutations of the same row set."
        )
    return ref_sort_idx[locations].astype(np.int64, copy=False)


def ordering_metrics(observed: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Compare an observed row order against a reference order.

    Both inputs are 1-D uint64 arrays (e.g., from :func:`collect_row_hashes`)
    and must be permutations of the same underlying row set.

    Returns a dict with:

    - ``n``: number of rows.
    - ``exact_match_fraction``: fraction of positions that coincide with the
      reference. 1.0 = identical order. 0.0 does *not* imply random; a
      uniform permutation gives ~1/N.
    - ``spearman_rho``: Spearman rank correlation between observed and
      reference positions. 1.0 = identical order, 0.0 ≈ uncorrelated,
      −1.0 = reversed.
    - ``displacement_score``: ``1 - mean(|Δrank|) / E_random[|Δrank|]``,
      clamped to ``[0, 1]``. 1.0 = identical; 0.0 means the mean positional
      shift matches a uniformly random permutation; between 0 and 1 means
      rows are on average *closer* to their reference position than random.
      This is the metric most aligned with the intuition that "more CPUs
      perturb locally but don't teleport rows to the end".
    - ``mean_rank_displacement``: average ``|observed_pos - reference_pos|``
      in raw rank units (handy for sanity-checking).
    """
    n = int(observed.shape[0])
    if n == 0:
        return {
            "n": 0,
            "exact_match_fraction": 1.0,
            "spearman_rho": 1.0,
            "displacement_score": 1.0,
            "mean_rank_displacement": 0.0,
        }

    obs_positions = _reference_positions(observed, reference)
    ideal = np.arange(n, dtype=np.int64)

    exact = float(np.mean(obs_positions == ideal))

    if n > 1:
        rho = float(
            np.corrcoef(obs_positions.astype(np.float64), ideal.astype(np.float64))[0, 1]
        )
    else:
        rho = 1.0

    displacement = np.abs(obs_positions - ideal)
    mean_disp = float(displacement.mean())
    # Expected mean |π(i) - i| for a uniformly random permutation of [0, n):
    #   (n^2 - 1) / (3n)  ≈ n/3.
    expected_random = (n * n - 1) / (3.0 * n) if n > 1 else 1.0
    displacement_score = (
        max(0.0, 1.0 - mean_disp / expected_random) if expected_random > 0 else 1.0
    )

    return {
        "n": n,
        "exact_match_fraction": exact,
        "spearman_rho": rho,
        "displacement_score": displacement_score,
        "mean_rank_displacement": mean_disp,
    }
