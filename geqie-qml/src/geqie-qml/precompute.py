import os
import numpy as np

from concurrent import futures
from multiprocessing import cpu_count
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Operator

import geqie
from geqie.encodings import frqi


# ---------------------------------------------------------------------------
# Module-level globals — set once per worker process by _init_worker
# ---------------------------------------------------------------------------

_sim = None
_U_INIT_CACHE = {}


# ---------------------------------------------------------------------------
# Worker initialiser — called once when each process in the pool starts
# ---------------------------------------------------------------------------

def _init_worker():
    """
    Per-process initialiser for the precompute pool.

    Pins the worker to a single OS thread and creates a module-level
    AerSimulator so it can be reused across all tasks in the same worker.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    global _sim
    _sim = AerSimulator(
        method="unitary",
        max_parallel_threads=1,
        max_parallel_shots=1,
        max_parallel_experiments=1,
    )


# ---------------------------------------------------------------------------
# Circuit encoding helpers — module-level for picklability
# ---------------------------------------------------------------------------

def _compute_circuit(image, geqie_encoding=frqi):
    """
    Encode a single image and return its full unitary matrix.

    The unitary is decomposed into a state-preparation part (cached per
    qubit count) and the encoding rotations, then multiplied together.

    Parameters
    ----------
    image : array-like
        Pixel data passed to ``geqie.encode``.
    geqie_encoding : module
        A geqie encoding module exposing ``init_function``, ``data_function``,
        and ``map_function``.  Defaults to FRQI.

    Returns
    -------
    np.ndarray, complex128, shape (2**n, 2**n)
    """
    global _sim
    qc = geqie.encode(
        geqie_encoding.init_function,
        geqie_encoding.data_function,
        geqie_encoding.map_function,
        image,
        perform_measurement=False,
    )
    qc_d = qc.decompose()
    n_qubits = qc_d.num_qubits

    rest_qc = QuantumCircuit(n_qubits)
    state_prep_qc = QuantumCircuit(n_qubits)

    for instr in qc_d.data:
        if instr.operation.name == "state_preparation":
            state_prep_qc.append(instr.operation, instr.qubits)
        if instr.operation.name not in ["reset", "measure", "barrier", "state_preparation"]:
            rest_qc.append(instr.operation, instr.qubits)

    rest_qc.save_unitary()
    U_rest = np.array(_sim.run(rest_qc).result().data()["unitary"])

    if n_qubits not in _U_INIT_CACHE:
        _U_INIT_CACHE[n_qubits] = Operator(state_prep_qc).data

    return U_rest @ _U_INIT_CACHE[n_qubits]


def _compute_save_single(args):
    """Single-sample worker — picklable at module level."""
    image, label, sample_number, save_dir, file_prefix = args
    filename = os.path.join(save_dir, f"{file_prefix}_{sample_number}")
    unitary_matrix = _compute_circuit(image)
    np.savez(filename, matrix=unitary_matrix, label=label, dtype=np.complex128)
    print(f"{filename} saved")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_and_save_circuits(
    data,
    labels,
    save_dir: str = "circuits",
    file_prefix: str = "matrix",
    number_of_workers: int | None = None,
):
    """
    Encode a dataset of images into unitary matrices and save them as .npz files.

    Each output file contains ``matrix`` (complex128 unitary) and ``label``
    (integer).  The files are consumed by ``MatrixDataset`` at training time.

    Parameters
    ----------
    data : array-like, shape (N, H, W)
        Images to encode.
    labels : array-like, shape (N,)
        Integer class labels, one per image.
    save_dir : str
        Directory where .npz files are written.  Created if absent.
    file_prefix : str
        Filename prefix; files are named ``{prefix}_{index}.npz``.
    number_of_workers : int | None
        Worker processes.  Defaults to (cpu_count - 1), min 1.
    """
    if number_of_workers is None:
        number_of_workers = max(1, cpu_count() - 1)

    os.makedirs(save_dir, exist_ok=True)

    args_list = [
        (data[i], labels[i], i, save_dir, file_prefix)
        for i in range(len(data))
    ]

    with futures.ProcessPoolExecutor(
        max_workers=number_of_workers,
        initializer=_init_worker,
    ) as executor:
        for _ in executor.map(_compute_save_single, args_list):
            pass
