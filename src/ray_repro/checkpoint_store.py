"""Filesystem-backed store for seen ``row_hash`` values, scoped per epoch.

Layout::

    {root}/current_epoch.txt
    {root}/execution_idx.txt
    {root}/epoch_{N}/segment_{K:04d}.parquet

Each call to :meth:`SeenHashStore.append` writes a new segment so checkpoints
within an epoch compose naturally (phase-1 writes segment 0, post-resume
writes segment 1, etc.). Reading reassembles all segments of an epoch into
a single uint64 array -- which is what a restart would do to rebuild the
filter for its current epoch.

Scoping by epoch directory makes "reset between epochs" trivial: either
``clear_epoch(N)`` or simply never read epoch ``M != current`` -- there is
no global "seen" state to unwind.

The top-level ``execution_idx.txt`` persists
:attr:`ray.data.DataContext._execution_idx` across process restarts. Ray's
counter is per-Dataset by design -- a rebuilt ``read_parquet(...)`` always
starts at 0 -- so if we want the reseed sequence to continue on resume, we
have to stash the counter here and restore it into the global
``DataContext`` before the next rebuild. See ``docs/execution-idx-semantics.md``.

``current_epoch.txt`` records the epoch index the pipeline is currently
processing. On a clean start it is ``0``; on each fully completed epoch it
is advanced to ``N+1``; on a crash mid-epoch it stays at ``N`` so a resuming
process knows to filter the already-seen rows out of epoch ``N`` rather
than treating it as fresh.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class SeenHashStore:
    """Per-epoch collection of consumed ``row_hash`` values on disk."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def epoch_dir(self, epoch: int) -> Path:
        return self.root / f"epoch_{epoch}"

    def append(self, epoch: int, hashes: np.ndarray) -> Path:
        """Persist ``hashes`` as a new segment under this epoch.

        Returns the path of the written segment. A no-op empty array still
        writes an (empty) segment so the segment counter advances, which
        keeps segment indices aligned with logical checkpoint events.
        """
        arr = np.asarray(hashes, dtype=np.uint64)
        d = self.epoch_dir(epoch)
        d.mkdir(parents=True, exist_ok=True)
        seg_idx = len(sorted(d.glob("segment_*.parquet")))
        path = d / f"segment_{seg_idx:04d}.parquet"
        table = pa.table({"row_hash": pa.array(arr, type=pa.uint64())})
        pq.write_table(table, path)
        return path

    def load_epoch(self, epoch: int) -> np.ndarray:
        """Return every seen row_hash for ``epoch`` as a uint64 array.

        Order is segment-by-segment, so the array reflects the order in
        which rows were appended. Tests that only care about set membership
        can simply wrap the result with ``set(...)``.
        """
        d = self.epoch_dir(epoch)
        if not d.exists():
            return np.empty(0, dtype=np.uint64)
        files = sorted(d.glob("segment_*.parquet"))
        if not files:
            return np.empty(0, dtype=np.uint64)
        arrs = [
            np.asarray(
                pq.read_table(f, columns=["row_hash"])["row_hash"],
                dtype=np.uint64,
            )
            for f in files
        ]
        return np.concatenate(arrs)

    def clear_epoch(self, epoch: int) -> None:
        """Remove all stored segments for ``epoch``."""
        d = self.epoch_dir(epoch)
        if d.exists():
            shutil.rmtree(d)

    def clear_all(self) -> None:
        """Remove every epoch's state. Fresh demo run starts from here."""
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Top-level counters (_execution_idx and current_epoch)
    #
    # Stored as plain-text integers so they are trivially inspectable and
    # survive schema changes to the parquet segments.
    # ------------------------------------------------------------------ #
    def _execution_idx_path(self) -> Path:
        return self.root / "execution_idx.txt"

    def load_execution_idx(self) -> int:
        """Return the persisted ``_execution_idx``, or ``0`` if absent."""
        return self._read_int(self._execution_idx_path())

    def save_execution_idx(self, idx: int) -> None:
        """Persist ``_execution_idx`` for the next rebuild / restart."""
        self._execution_idx_path().write_text(f"{int(idx)}\n")

    def _current_epoch_path(self) -> Path:
        return self.root / "current_epoch.txt"

    def load_current_epoch(self) -> int:
        """Return the epoch to resume work on, or ``0`` if absent.

        A freshly cleared store, or a store that has never been written to,
        returns ``0`` -- the canonical "start from the beginning" state.
        """
        return self._read_int(self._current_epoch_path())

    def save_current_epoch(self, epoch: int) -> None:
        """Persist the current epoch index for the next rebuild / restart."""
        self._current_epoch_path().write_text(f"{int(epoch)}\n")

    @staticmethod
    def _read_int(path: Path) -> int:
        if not path.exists():
            return 0
        text = path.read_text().strip()
        return int(text) if text else 0
