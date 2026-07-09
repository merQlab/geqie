import importlib
import logging
import os

from typing import Any
from types import ModuleType

import numpy as np
from concurrent import futures
from multiprocessing import cpu_count
from tqdm import tqdm

from qiskit.quantum_info import Operator

import geqie

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_and_save_circuits(
    data,
    labels,
    save_dir: str = "circuits",
    file_prefix: str = "matrix",
    number_of_workers: int | None = None,
    geqie_encoding: str | ModuleType = "frqi",
    encoding_params: dict[str, Any] = {},
):
    """
    Encode a dataset of images into unitary matrices and save them as .npz files.

    Each output file contains ``matrix`` (complex128 unitary) and ``label``
    (integer).  Files are named ``{prefix}_{index}_label_{label}.npz`` and can
    be loaded at training time via :class:`ZipMatrixDataset`.

    Parameters
    ----------
    data : array-like, shape (N, H, W)
        Images to encode.
    labels : array-like, shape (N,)
        Integer class labels, one per image.
    save_dir : str
        Directory where .npz files are written.  Created if absent.
    file_prefix : str
        Filename prefix; files are named ``{prefix}_{index}_label_{label}.npz``.
    number_of_workers : int | None
        Number of worker processes.  Defaults to ``cpu_count - 1``, minimum 1.
    geqie_encoding : str
        GEQIE encoding name, e.g. ``"frqi"``.
    encoding_params : dict
        Additional keyword arguments forwarded to the encoding function.
    """
    if number_of_workers is None:
        number_of_workers = max(1, cpu_count() - 1)

    os.makedirs(save_dir, exist_ok=True)
    encoding_name = _normalize_encoding_name(geqie_encoding)

    logger.debug(f"Starting precompute with {number_of_workers} workers for encoding '{encoding_name}'")
    if number_of_workers == 1:
        for i in tqdm(range(len(data)), total=len(data)):
            _compute_save_single(
                image=data[i],
                label=labels[i],
                sample_index=i,
                save_dir=save_dir,
                file_prefix=file_prefix,
                geqie_encoding=encoding_name,
                encoding_params=encoding_params
            )
    else:
        with futures.ProcessPoolExecutor(max_workers=number_of_workers) as executor:
            precompute_futures = [
                executor.submit(
                    _compute_save_single,
                    image=data[i],
                    label=labels[i],
                    sample_index=i,
                    save_dir=save_dir,
                    file_prefix=file_prefix,
                    geqie_encoding=encoding_name,
                    encoding_params=encoding_params,
                ) for i in tqdm(range(len(data)), total=len(data), desc="Submitting tasks")
            ]

            for future in tqdm(futures.as_completed(precompute_futures), total=len(precompute_futures), desc="Processing tasks"):
                future.result()


# ---------------------------------------------------------------------------
# Worker initialiser — called once when each process in the pool starts
# ---------------------------------------------------------------------------

def _init_worker():
    """
    Per-process initializer for the precompute pool.

    Pins each worker to a single OS thread to prevent thread over-subscription
    when many worker processes run in parallel.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


# ---------------------------------------------------------------------------
# Circuit encoding helpers
# ---------------------------------------------------------------------------

def _normalize_encoding_name(geqie_encoding: str) -> str:
    """Return a lower-cased, stable encoding key from the given string."""
    if not isinstance(geqie_encoding, str):
        raise TypeError(f"geqie_encoding must be a string got {type(geqie_encoding).__name__}.")        
    
    return geqie_encoding.lower()


def _import_encoding_module(encoding_name: str):
    """Import and return the ``geqie.encodings.<name>`` module for the given encoding key."""
    normalized_name = _normalize_encoding_name(encoding_name)
    return importlib.import_module(f"geqie.encodings.{normalized_name}")


def _compute_circuit_unitary(image, geqie_encoding: str = "frqi", encoding_params: dict[str, Any] = {}):
    """
    Encode a single image and return its full unitary matrix.

    Parameters
    ----------
    image : array-like
        Pixel data passed to ``geqie.encode``.
    geqie_encoding : str
        GEQIE encoding name, e.g. ``"frqi"``.
    encoding_params : dict
        Additional keyword arguments forwarded to the encoding function.

    Returns
    -------
    numpy.ndarray
        Complex128 array of shape ``(2**n, 2**n)``.
    """
    encoding_module = _import_encoding_module(_normalize_encoding_name(geqie_encoding))
    circuit = geqie.encode(
        encoding_module.init_function,
        encoding_module.data_function,
        encoding_module.map_function,
        image=image,
        perform_measurement=False,
        encoding_params=encoding_params,
    )
    return Operator.from_circuit(circuit).to_matrix()


def _compute_save_single(image, label, sample_index, save_dir, file_prefix, geqie_encoding, encoding_params):
    """Encode one image and save its unitary matrix to a .npz file."""
    filename = os.path.join(save_dir, f"{file_prefix}_{sample_index}_label_{label}")
    unitary_matrix = _compute_circuit_unitary(image, geqie_encoding, encoding_params)
    np.savez(file=filename, matrix=unitary_matrix, label=label, dtype=np.complex128)
