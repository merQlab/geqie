import os
import numpy as np
import torch
import torch.nn as nn

from contextlib import contextmanager
from concurrent import futures
from multiprocessing import cpu_count
from torch.utils.data import Dataset
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.gradients import SPSASamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MatrixDataset(Dataset):
    """
    PyTorch Dataset that reads pre-computed unitary matrices from .npz files.

    Each file must contain:
      - ``matrix``: complex128 array of shape (2**n, 2**n)
      - ``label``:  integer class label

    These files are produced by :func:`precompute.compute_and_save_circuits`.
    """

    def __init__(self, file_paths):
        self.files = file_paths

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        matrix = torch.tensor(data["matrix"], dtype=torch.complex128)
        label = torch.tensor(data["label"], dtype=torch.long)
        return matrix, label


# ---------------------------------------------------------------------------
# Module-level picklable helpers for ProcessPoolExecutor workers
#
# These MUST live at module level (not inside a class) so that Python's
# multiprocessing "spawn" start-method can pickle them.
# ---------------------------------------------------------------------------

def _build_vqc_circuit(num_qubits: int, num_layers: int):
    """Reconstruct the parameterised VQC (without the image unitary)."""
    thetas = ParameterVector("theta", length=3 * num_qubits * num_layers)
    vqc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        offset = layer * 3 * num_qubits
        for i in range(num_qubits):
            vqc.rx(thetas[offset + i], i)
        for i in range(num_qubits):
            vqc.ry(thetas[offset + num_qubits + i], i)
        for i in range(num_qubits):
            vqc.rz(thetas[offset + 2 * num_qubits + i], i)
        # Alternating brickwork entanglement
        if layer % 2 == 0:
            for i in range(0, num_qubits - 1, 2):
                vqc.cx(i, i + 1)
        else:
            for i in range(1, num_qubits - 1, 2):
                vqc.cx(i, i + 1)
            vqc.cx(num_qubits - 1, 0)
    return vqc


def _worker_forward_eval(args):
    """
    Forward pass for a single sample.  Called inside a worker process.

    Returns
    -------
    np.ndarray, shape (2**num_qubits,)
        Probability distribution over basis states.
    """
    matrix_np, weights_np, num_qubits, num_layers, shots = args
    vqc = _build_vqc_circuit(num_qubits, num_layers)
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(matrix_np), range(num_qubits))
    qc.compose(vqc, inplace=True)

    sampler = Sampler(default_shots=shots)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=None,
        weight_params=list(qc.parameters),
        sampler=sampler,
        gradient=SPSASamplerGradient(sampler=sampler),
    )
    return qnn.forward(input_data=None, weights=weights_np).flatten()


def _worker_grad_eval(args):
    """
    SPSA gradient for a single sample.  Called inside a worker process.

    Returns
    -------
    np.ndarray, shape (output_size, num_weights)
        Jacobian of the output probabilities w.r.t. the quantum weights.
    """
    matrix_np, weights_np, num_qubits, num_layers, shots = args
    vqc = _build_vqc_circuit(num_qubits, num_layers)
    qc = QuantumCircuit(num_qubits)
    qc.append(UnitaryGate(matrix_np), range(num_qubits))
    qc.compose(vqc, inplace=True)

    sampler = Sampler(default_shots=shots)
    grad_fn = SPSASamplerGradient(sampler=sampler)
    qnn = SamplerQNN(
        circuit=qc,
        input_params=None,
        weight_params=list(qc.parameters),
        sampler=sampler,
        gradient=grad_fn,
    )
    # SamplerQNN.backward → (input_grad, weight_grad)
    # weight_grad shape: (1, output_size, num_weights) for a single sample
    _, weight_grad = qnn.backward(input_data=None, weights=weights_np)
    return weight_grad.squeeze(0)  # (output_size, num_weights)


def _init_training_worker():
    """
    Per-worker initialiser: pin each worker to a single OS thread so that
    N workers each use 1 core and together fill all available cores, rather
    than every worker spawning its own thread-pool and over-subscribing.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _work_dispatch(fn, args_list, executor):
    """
    Dispatch per-sample work either sequentially or in parallel.

    When executor is None, calls fn(args) in a plain loop on the current
    process — identical computation to the parallel path, just no pool.
    When executor is a ProcessPoolExecutor, submits all jobs simultaneously
    and collects results.

    Parameters
    ----------
    fn        : callable — _worker_forward_eval or _worker_grad_eval
    args_list : list of arg tuples, one per sample
    executor  : ProcessPoolExecutor | None

    Returns
    -------
    list of results, one per sample, in input order
    """
    if executor is None:
        return [fn(args) for args in args_list]
    jobs = [executor.submit(fn, args) for args in args_list]
    return [f.result() for f in jobs]


# ---------------------------------------------------------------------------
# Custom autograd Function — unified sequential and parallel paths
# ---------------------------------------------------------------------------

class QNNBatchFunction(torch.autograd.Function):
    """
    Custom autograd Function that evaluates a batch of quantum circuits and
    computes parameter-shift gradients.

    Both the forward and backward passes are built on the same per-sample
    worker functions (_worker_forward_eval, _worker_grad_eval).  The only
    difference between sequential and parallel execution is whether those
    functions are called in a loop or dispatched to a ProcessPoolExecutor —
    controlled by the ``executor`` argument.

    Parameters (positional, as required by torch.autograd.Function)
    ----------
    weights           : (num_params,) tensor — the trainable quantum weights
    executor          : ProcessPoolExecutor | None
                        None  → sequential evaluation on the calling process
                        Pool  → parallel evaluation across worker processes
    matrices_np_list  : list of np.ndarray, each (2^n, 2^n) image unitary
    num_qubits        : int
    num_layers        : int
    shots             : int
    """

    @staticmethod
    def forward(ctx, weights, executor, matrices_np_list,
                num_qubits, num_layers, shots):
        weights_np = weights.detach().numpy()

        args_list = [
            (matrix, weights_np, num_qubits, num_layers, shots)
            for matrix in matrices_np_list
        ]
        probs_list = _work_dispatch(_worker_forward_eval, args_list, executor)
        probs_np = np.stack(probs_list)  # (batch_size, output_size)

        ctx.save_for_backward(weights)
        ctx.executor = executor
        ctx.matrices_np_list = matrices_np_list
        ctx.num_qubits = num_qubits
        ctx.num_layers = num_layers
        ctx.shots = shots

        return torch.tensor(probs_np, dtype=weights.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output : (batch_size, output_size) — upstream gradient

        Returns one gradient tensor per positional argument of forward().
        Non-differentiable arguments return None.
        """
        weights, = ctx.saved_tensors
        weights_np = weights.detach().numpy()

        args_list = [
            (m, weights_np, ctx.num_qubits, ctx.num_layers, ctx.shots)
            for m in ctx.matrices_np_list
        ]
        jac_list = _work_dispatch(_worker_grad_eval, args_list, ctx.executor)

        jac_np = np.stack(jac_list)             # (batch, output, num_weights)
        grad_np = grad_output.detach().numpy()  # (batch, output)

        # Chain rule: accumulate over batch and output dimensions
        # d_loss/d_w = sum_{b,o} (d_loss/d_prob_{b,o}) * (d_prob_{b,o}/d_w)
        weight_grad_np = np.einsum("bo,bop->p", grad_np, jac_np)

        return (
            torch.tensor(weight_grad_np, dtype=weights.dtype),  # weights
            None,   # executor         — not differentiable
            None,   # matrices_np_list — not differentiable
            None,   # num_qubits
            None,   # num_layers
            None,   # shots
        )


# ---------------------------------------------------------------------------
# VQCLayer — a composable PyTorch layer
# ---------------------------------------------------------------------------

class VQCLayer(nn.Module):
    """
    Variational Quantum Circuit layer, usable as a standard PyTorch nn.Module.

    Accepts a batch of pre-encoded unitary matrices and returns a batch of
    probability vectors over all 2**num_qubits basis states.  Classical
    post-processing (linear head, activation, loss) is left to the caller,
    so this layer composes freely inside any nn.Sequential or custom Module.

    Both the sequential (default) and parallel paths are driven by the same
    per-sample worker functions — ``_worker_forward_eval`` for the forward
    pass and ``_worker_grad_eval`` for the parameter-shift backward pass.
    The only difference is whether those functions are called in a plain loop
    or dispatched to a ``ProcessPoolExecutor`` via ``parallel_context()``.

    Parameters
    ----------
    num_qubits : int
        Number of qubits.  Input matrices must be square with side 2**num_qubits.
    num_layers : int
        Number of brickwork VQC layers (each with Rx/Ry/Rz + entanglement).
    shots : int
        Number of measurement shots per circuit evaluation.
    scale_output : bool
        When True (default), multiply output probabilities by 2**num_qubits.
        This rescales the near-zero probability values into a more numerically
        convenient range before they are passed to a classical head.

    Usage
    -----
    Minimal classification network::

        model = nn.Sequential(
            VQCLayer(num_qubits=9, num_layers=1),
            nn.Linear(2**9, num_classes),
            nn.LogSoftmax(dim=-1),
        )

    Training with parallel circuit evaluation::

        vqc = VQCLayer(num_qubits=9, num_layers=1)
        model = nn.Sequential(vqc, nn.Linear(2**9, num_classes), nn.LogSoftmax(dim=-1))

        with vqc.parallel_context(num_workers=8):
            for epoch in range(epochs):
                for batch_matrices, batch_labels in dataloader:
                    ...
    """

    def __init__(
        self,
        num_qubits: int = 9,
        num_layers: int = 3,
        shots: int = 1024,
        scale_output: bool = True,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_shots = shots
        self.scale_output = scale_output

        # Trainable quantum weights, registered as a proper nn.Parameter so
        # that optimisers, state_dict, and requires_grad all work out of the box.
        # Workers reconstruct the circuit independently via _build_vqc_circuit,
        # so the Qiskit QuantumCircuit object does not need to be stored here.
        num_params = 3 * num_qubits * num_layers
        self.quantum_weight = nn.Parameter(
            torch.empty(num_params).uniform_(-np.pi, np.pi)
        )

        # Executor is None by default; populated only inside parallel_context().
        self._executor: futures.ProcessPoolExecutor | None = None

    @contextmanager
    def parallel_context(self, num_workers: int | None = None):
        """
        Context manager that starts a process pool and enables parallel
        circuit evaluation for the duration of the ``with`` block.

        Without this context, forward() and backward() run sequentially on
        the calling process using the same underlying per-sample functions.

        Parameters
        ----------
        num_workers : int | None
            Number of worker processes.  Defaults to (cpu_count - 1), min 1.

        Example
        -------
        ::

            with vqc_layer.parallel_context(num_workers=8):
                for epoch in range(epochs):
                    for batch, labels in dataloader:
                        ...
        """
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        mp_ctx = __import__("multiprocessing").get_context("spawn")
        with futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_training_worker,
            mp_context=mp_ctx,
        ) as executor:
            self._executor = executor
            try:
                yield
            finally:
                # Always clear the reference, even if the training loop raises.
                self._executor = None

    def forward(self, batched_matrices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        batched_matrices : torch.Tensor, shape (batch_size, 2**n, 2**n), complex
            Batch of pre-encoded image unitary matrices.

        Returns
        -------
        torch.Tensor, shape (batch_size, 2**num_qubits), float
            Probability distribution over basis states for each sample.
            If ``scale_output=True``, values are multiplied by 2**num_qubits.
        """
        dim = 2 ** self.num_qubits
        if batched_matrices.shape[1:] != torch.Size([dim, dim]):
            raise ValueError(
                f"VQCLayer expects unitary matrices of shape {(dim, dim)}, "
                f"but got input with shape {tuple(batched_matrices.shape[1:])}. "
                f"Check that num_qubits={self.num_qubits} matches your encoded data."
            )

        matrices_np_list = [m.detach().cpu().numpy() for m in batched_matrices]

        # QNNBatchFunction runs sequentially when self._executor is None,
        # or fans out to the process pool when parallel_context() is active.
        probs = QNNBatchFunction.apply(
            self.quantum_weight,
            self._executor,
            matrices_np_list,
            self.num_qubits,
            self.num_layers,
            self.num_shots,
        )

        if self.scale_output:
            probs = probs * dim

        return probs

    def extra_repr(self) -> str:
        """Adds layer details to the standard nn.Module string representation."""
        return (
            f"num_qubits={self.num_qubits}, num_layers={self.num_layers}, "
            f"shots={self.num_shots}, scale_output={self.scale_output}, "
            f"num_params={self.quantum_weight.numel()}"
        )
