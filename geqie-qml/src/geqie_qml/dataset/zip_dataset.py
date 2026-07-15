import io
import logging
import re
import threading
import weakref
import zipfile

from collections import OrderedDict
from typing import Dict, List

import numpy as np
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)

_FILENAME_RE = re.compile(r"^matrix_(\d+)_label_(\w+)\.npz$")


# ---------------------------------------------------------------------------
# Lazy zip-based dataset
# ---------------------------------------------------------------------------

class ZipMatrixDataset(Dataset):
    """
    PyTorch Dataset that lazily reads pre-computed unitary matrices from .npz
    files stored inside a zip archive.

    The archive may nest ``train`` and ``test`` folders at any depth; the
    loader searches the zip tree automatically to locate them.

    Expected zip layout::

        archive.precomputed.zip
        ├── train/
        │   ├── matrix_0_label_7.npz
        │   └── ...
        └── test/
            ├── matrix_0_label_2.npz
            └── ...

    Each .npz file must contain:

    - ``matrix``: complex128 array of shape ``(2**n, 2**n)``
    - ``label``:  class label (can be string)

    A thread-local ``ZipFile`` handle is cached per worker so that the zip
    central directory is parsed only once per process/thread, making the
    dataset safe and efficient with multi-worker ``DataLoader`` usage.
    """

    def __init__(self, zip_path: str, split_name: str = "train", cache_size: int = 0):
        """
        Parameters
        ----------
        zip_path : str
            Path to the ``.precomputed.zip`` archive.
        split_name : str
            Dataset split name from zip to load (e.g. ``"train"``, ``"test"``).
        cache_size : int
            Maximum number of decoded ``(matrix, label)`` items to keep in an
            in-memory LRU cache. ``0`` (the default) disables caching, which
            keeps memory bounded and predictable for large matrices. A positive
            value trades memory for speed by retaining the most recently used
            items; note the cache lives *per DataLoader worker process*.
        """
        self._zip_path = zip_path
        self._split_name = split_name
        self._cache_size = max(0, int(cache_size))
        self._entry_index = self._index_zip(zip_path, split_name)
        # LRU cache of loaded matrices; empty/unused when cache_size == 0.
        self._cache: OrderedDict[int, tuple] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._local = threading.local()  # per-thread ZipFile handle
        self._open_handles: list[zipfile.ZipFile] = []
        self._handles_lock = threading.Lock()
        # Fires on GC and at interpreter exit via atexit — covers normal
        # shutdown and KeyboardInterrupt.  Hard kills (SIGKILL) cannot be
        # caught at the Python level; the OS reclaims the file descriptors.
        weakref.finalize(self, ZipMatrixDataset._close_handles,
                         self._open_handles, self._handles_lock)

    # ------------------------------------------------------------------
    # Pickle support (required for multi-process DataLoader on Windows/spawn)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # threading.local and Lock are not picklable; drop them so the object
        # can be sent to worker processes.  __setstate__ recreates them.
        del state["_local"]
        del state["_open_handles"]
        del state["_handles_lock"]
        del state["_cache_lock"]
        # Start each worker with a fresh cache instead of shipping loaded
        # matrices across the process boundary.
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._cache_lock = threading.Lock()
        self._local = threading.local()
        self._open_handles: list[zipfile.ZipFile] = []
        self._handles_lock = threading.Lock()
        weakref.finalize(self, ZipMatrixDataset._close_handles,
                         self._open_handles, self._handles_lock)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_split_prefix(names: list[str], split_name: str) -> str | None:
        """
        Search the normalized zip entry tree for the shallowest directory that
        contains the specified split folder (e.g. ``"train"``, ``"test"``), then return the
        full prefix that leads to *split* (e.g. ``"some/nested/train/"``).

        Returns ``None`` if no qualifying directory is found.
        """
        # Collect every unique parent path of every entry, normalized.
        parent_dirs: set[str] = set()
        for name in names:
            norm = name.replace("\\", "/")
            parts = [p for p in norm.split("/") if p]
            # Each prefix level is a candidate parent.
            for depth in range(len(parts)):
                parent_dirs.add("/".join(parts[:depth]))  # may be "" for root

        for parent in sorted(parent_dirs, key=lambda p: p.count("/")):
            split_prefix = (parent + "/" if parent else "") + split_name + "/"
            has_split = any(
                n.replace("\\", "/").startswith(split_prefix) for n in names
            )
            if has_split:
                return split_prefix

        return None

    @staticmethod
    def _index_zip(zip_path: str, split_name: str) -> Dict[int, tuple[str, str]]:
        """Return a dict of ``index: (zip_member_name, label)`` pairs."""
        entries: Dict[int, tuple[str, str]] = {}  # (index, (member, label))

        with zipfile.ZipFile(zip_path, "r") as zf:
            all_names = zf.namelist()
            prefix = ZipMatrixDataset._find_split_prefix(all_names, split_name)

            if prefix is None:
                import warnings
                warnings.warn(
                    f"Could not find a '{split_name}/' directory containing '*.npz' files "
                    f"in '{zip_path}'. First entries: {all_names[:10]}",
                    UserWarning,
                    stacklevel=3,
                )
                return {}

            for name in all_names:
                norm = name.replace("\\", "/")
                if not norm.startswith(prefix):
                    continue
                basename = norm[len(prefix):]
                # Only direct children — skip deeper nesting.
                if "/" in basename:
                    continue
                m = _FILENAME_RE.match(basename)
                if m is None:
                    continue
                idx, label = int(m.group(1)), m.group(2)
                entries[idx] = (name, label)

        return entries

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _close_handles(
        handles: "list[zipfile.ZipFile]", lock: threading.Lock
    ) -> None:
        """Close all tracked ZipFile handles.  Called by ``weakref.finalize``."""
        with lock:
            for zf in handles:
                try:
                    zf.close()
                except Exception:
                    pass
            handles.clear()

    def _get_zip(self) -> zipfile.ZipFile:
        """Return a cached, thread-local (and process-local) ZipFile handle.

        Opening the zip on every ``__getitem__`` call forces Python to parse the
        entire central directory each time, which is expensive and causes
        ``MemoryError`` when many DataLoader workers do it simultaneously.
        Caching the handle per-thread avoids that overhead while remaining safe
        for multi-threaded and multi-process DataLoader workers.
        """
        if not hasattr(self._local, "zf"):
            zf = zipfile.ZipFile(self._zip_path, "r")
            self._local.zf = zf
            with self._handles_lock:
                self._open_handles.append(zf)
        return self._local.zf

    def __len__(self) -> int:
        return len(self._entry_index)

    def __getitem__(self, idx: int):
        import torch

        member, label = self._entry_index[idx]

        if self._cache_size > 0:
            with self._cache_lock:
                if idx in self._cache:
                    # Mark as most-recently-used.
                    self._cache.move_to_end(idx)
                    return self._cache[idx]

        with self._get_zip().open(member) as f:
            buf = io.BytesIO(f.read())
        data = np.load(buf)
        matrix = torch.tensor(data["matrix"], dtype=torch.complex128)
        label_tensor = torch.tensor(data["label"], dtype=torch.long)
        result = (matrix, label_tensor)

        if self._cache_size > 0:
            with self._cache_lock:
                self._cache[idx] = result
                self._cache.move_to_end(idx)
                # Evict least-recently-used items beyond the budget.
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)

        return result


def load_precomputed_zip_matrices(
    zip_path: str,
    split_names: List[str] = ["train", "test"],
    cache_size: int = 0,
) -> tuple[ZipMatrixDataset, ...]:
    """
    Load train and test splits from a ``.precomputed.zip`` archive.

    The function searches the zip tree for ``split_names`` folders at
    any nesting depth and returns lazy :class:`ZipMatrixDataset` objects for each split.

    Parameters
    ----------
    zip_path : str
        Path to the zip archive produced by :func:`compute_and_save_circuits`.
    cache_size : int
        Maximum number of decoded items each split may keep in its in-memory
        LRU cache. ``0`` (the default) disables caching to keep memory bounded
        for large matrices.

    Returns
    -------
    datasets : tuple[ZipMatrixDataset, ...]
        Lazy datasets for each split.
    """
    datasets = []
    for split_name in split_names:
        datasets.append(
            ZipMatrixDataset(zip_path, split_name=split_name, cache_size=cache_size)
        )
    return tuple(datasets)
