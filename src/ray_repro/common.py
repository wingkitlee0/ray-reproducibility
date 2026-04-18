"""Shared pipeline + fingerprint helpers for the reproducibility examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import ray
import xxhash
from ray.data import Dataset, FileShuffleConfig, RandomSeedConfig


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


def collect_row_hashes(batches: Iterable) -> np.ndarray:
    """Materialize the per-epoch ``row_hash`` sequence as a 1-D uint64 array.

    Order matches the order rows are yielded by the iterator -- i.e., the
    exact ingestion order a training loop would see. Unlike
    :func:`fingerprint_epoch`, this preserves enough information to compute
    continuous ordering metrics against a reference sequence.
    """
    chunks: list[np.ndarray] = []
    for batch in batches:
        chunks.append(np.asarray(batch["row_hash"], dtype=np.uint64))
    if not chunks:
        return np.empty(0, dtype=np.uint64)
    return np.concatenate(chunks)


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
