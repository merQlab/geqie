import time
import os
import glob
import torch
import numpy as np
import torch.nn as nn

from contextlib import contextmanager
from PIL import Image
from concurrent import futures
from multiprocessing import cpu_count
from datasets import load_dataset
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.gradients import SPSASamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors.torch_connector import _TorchNNFunction

import geqie
from geqie.encodings import frqi


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MatrixDataset(Dataset):
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
        gradient=SPSASamplerGradient(sampler=sampler)
    )
    return qnn.forward(input_data=None, weights=weights_np).flatten()


def _worker_grad_eval(args):
    """
    Parameter-shift gradient for a single sample.  Called inside a worker.

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


# ---------------------------------------------------------------------------
# Custom autograd Function — the core of the parallelisation
# ---------------------------------------------------------------------------

class ParallelQNNBatchFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weights, executor, matrices_np_list,
                num_qubits, num_layers, shots):
        """
        Parameters
        ----------
        weights           : (num_params,) tensor — the trainable quantum weights
        executor          : ProcessPoolExecutor shared across batches
        matrices_np_list  : list of np.ndarray, each (2^n, 2^n) image unitary
        num_qubits        : int
        num_layers        : int
        shots             : int | None
        """
        weights_np = weights.detach().numpy()

        # --- Submit all forward evaluations simultaneously ---
        args_list = [
            (matrix, weights_np, num_qubits, num_layers, shots)
            for matrix in matrices_np_list
        ]
        jobs_list = [executor.submit(_worker_forward_eval, a) for a in args_list]
        probs_list = [f.result() for f in jobs_list]  # blocks until all done

        probs_np = np.stack(probs_list)  # (batch_size, output_size)

        # Save context for backward
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

        # --- Submit all gradient evaluations simultaneously ---
        args_list = [
            (m, weights_np, ctx.num_qubits, ctx.num_layers, ctx.shots)
            for m in ctx.matrices_np_list
        ]
        futs = [ctx.executor.submit(_worker_grad_eval, a) for a in args_list]
        # Each result: (output_size, num_weights)
        jac_list = [f.result() for f in futs]

        jac_np = np.stack(jac_list)             # (batch, output, num_weights)
        grad_np = grad_output.detach().numpy()  # (batch, output)

        # Chain rule: accumulate over batch and output dimensions
        # d_loss/d_w = sum_{b,o} (d_loss/d_prob_{b,o}) * (d_prob_{b,o}/d_w)
        weight_grad_np = np.einsum("bo,bop->p", grad_np, jac_np)

        return (
            torch.tensor(weight_grad_np, dtype=weights.dtype),  # weights
            None,   # executor        — not differentiable
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

        # Build the parameterised VQC once; workers reconstruct it independently
        # via _build_vqc_circuit so that they don't need to pickle this object.
        self.vqc_circuit = _build_vqc_circuit(num_qubits, num_layers)

        # Trainable quantum weights, registered as a proper nn.Parameter so
        # that optimisers, state_dict, and requires_grad all work out of the box.
        num_params = 3 * num_qubits * num_layers
        self.quantum_weight = nn.Parameter(
            torch.empty(num_params).uniform_(-np.pi, np.pi)
        )

        # Executor is None by default; set only while inside parallel_context().
        self._executor: futures.ProcessPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Parallel context manager — keeps executor lifecycle self-contained
    # ------------------------------------------------------------------

    @contextmanager
    def parallel_context(self, num_workers: int | None = None):
        """
        Context manager that starts a process pool and enables the fast
        parallel forward/backward path for the duration of the ``with`` block.

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

    # ------------------------------------------------------------------
    # Sequential forward (fallback when no parallel context is active)
    # ------------------------------------------------------------------

    def _forward_sequential(self, batched_matrices):
        batched_probs = []
        for single_matrix_tensor in batched_matrices:
            matrix_np = single_matrix_tensor.detach().cpu().numpy()
            qc = QuantumCircuit(self.num_qubits)
            qc.append(UnitaryGate(matrix_np), range(self.num_qubits))
            qc.compose(self.vqc_circuit, inplace=True)

            fresh_sampler = Sampler(default_shots=self.num_shots)
            fresh_qnn = SamplerQNN(
                circuit=qc,
                input_params=None,
                weight_params=list(qc.parameters),
                sampler=fresh_sampler,
                gradient=SPSASamplerGradient(sampler=fresh_sampler)
            )
            probs = _TorchNNFunction.apply(
                torch.empty(0),
                self.quantum_weight,
                fresh_qnn,
                False
            )
            batched_probs.append(probs)

        return torch.stack(batched_probs)

    # ------------------------------------------------------------------
    # Parallel forward — uses ParallelQNNBatchFunction
    # ------------------------------------------------------------------

    def _forward_parallel(self, batched_matrices):
        matrices_np_list = [
            m.detach().cpu().numpy() for m in batched_matrices
        ]
        return ParallelQNNBatchFunction.apply(
            self.quantum_weight,
            self._executor,
            matrices_np_list,
            self.num_qubits,
            self.num_layers,
            self.num_shots,
        )

    # ------------------------------------------------------------------
    # forward — validates input shape, dispatches, applies optional scaling
    # ------------------------------------------------------------------

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
        expected_shape = torch.Size([dim, dim])
        if batched_matrices.shape[1:] != expected_shape:
            raise ValueError(
                f"VQCLayer expects unitary matrices of shape {(dim, dim)}, "
                f"but got input with shape {tuple(batched_matrices.shape[1:])}. "
                f"Check that num_qubits={self.num_qubits} matches your encoded data."
            )

        if self._executor is not None:
            probs = self._forward_parallel(batched_matrices)
        else:
            probs = self._forward_sequential(batched_matrices)

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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_QCNN(
    epochs: int = 20,
    num_classes: int = 2,
    batch_size: int = 5,
    num_qubits: int = 9,
    num_layers: int = 1,
    circuits_directory: str = "circuits",
    num_workers: int | None = None,
):
    """
    Train a VQCLayer followed by a classical linear head.

    The model is assembled here to illustrate the intended compositional
    pattern — VQCLayer produces probability features, the linear head maps
    them to class logits, and LogSoftmax + NLLLoss combine as cross-entropy.

    Parameters
    ----------
    epochs, num_classes, batch_size : int
    num_qubits : int
        Must match the unitary matrices stored in circuits_directory.
    num_layers : int
        Number of brickwork VQC layers.
    circuits_directory : str
        Directory containing .npz files produced by compute_and_save_circuits.
    num_workers : int | None
        Worker processes for parallel simulation.  Defaults to cpu_count - 1.
    """
    # --- Build model from composable parts ---
    vqc_layer = VQCLayer(num_qubits=num_qubits, num_layers=num_layers)
    model = nn.Sequential(
        vqc_layer,
        nn.Linear(2 ** num_qubits, num_classes),
        nn.LogSoftmax(dim=-1),
    )

    optimizer = Adam([
        {"params": vqc_layer.quantum_weight,          "lr": 0.001},
        {"params": model[1].parameters(),             "lr": 0.01},
    ])

    f_loss = NLLLoss()
    model.train()
    loss_list = []

    files = sorted(glob.glob(f"{circuits_directory}/*.npz"))
    print(f"Found {len(files)} circuit files")
    dataset = MatrixDataset(files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    print(f"Starting training with {num_workers} worker processes")

    # The parallel_context() block starts and owns the process pool.
    # No manual executor injection needed — vqc_layer handles it internally.
    with vqc_layer.parallel_context(num_workers=num_workers):
        for epoch in range(epochs):
            start = time.time()
            total_loss = []

            for batch_matrices, batch_labels in dataloader:
                optimizer.zero_grad()

                output = model(batch_matrices)
                loss = f_loss(output, batch_labels)
                loss.backward()

                total_loss.append(loss.item())
                optimizer.step()

            epoch_loss = sum(total_loss) / len(total_loss)
            loss_list.append(epoch_loss)
            elapsed = time.time() - start
            print(
                f"Epoch {epoch + 1}/{epochs}  "
                f"Loss: {epoch_loss:.4f}  "
                f"Time: {elapsed:.2f}s"
            )

    return loss_list, model


# ---------------------------------------------------------------------------
# Data loading & circuit precomputation
# ---------------------------------------------------------------------------

def load_and_process_mnist_dataset(
    path_to_mnist_dataset,
    labels_to_include=[0, 1],
    n_samples_per_label=100,
    resize=(8, 8),
):
    mnist_dataset = load_dataset("parquet", data_files=path_to_mnist_dataset)

    selected_images = {x: [] for x in labels_to_include}
    for image_idx in range(len(mnist_dataset["train"])):
        label = mnist_dataset["train"][image_idx]["label"]
        if label in labels_to_include:
            if len(selected_images[label]) < n_samples_per_label:
                resized_image = mnist_dataset["train"][image_idx]["image"].resize(
                    resize, resample=Image.BILINEAR
                )
                selected_images[label].append({
                    "image": np.array(resized_image),
                    "label": label
                })

    selected_images = [item for sublist in selected_images.values() for item in sublist]

    X = np.array([item["image"] for item in selected_images])
    y = np.array([item["label"] for item in selected_images])
    return X, y


_U_INIT_CACHE = {}


def _compute_circuit(image, geqie_encoding=frqi):
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


def _init_worker():
    """Called once when each precompute worker process starts."""
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


def _compute_save_single(args):
    """Single-sample worker — picklable at module level."""
    image, label, sample_number, save_dir, file_prefix = args
    filename = os.path.join(save_dir, f"{file_prefix}_{sample_number}")
    unitary_matrix = _compute_circuit(image)
    np.savez(filename, matrix=unitary_matrix, label=label, dtype=np.complex128)
    print(f"{filename} saved")


def compute_and_save_circuits(
    data,
    labels,
    save_dir="new_pool_circuits",
    file_prefix="matrix",
    number_of_workers=None,
):
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


if __name__ == "__main__":
    current_dir = os.getcwd()
    mnist_dataset = os.path.join("train-00000-of-00001.parquet")
    path_to_mnist_dataset = os.path.join(current_dir, mnist_dataset)
    X, y = load_and_process_mnist_dataset(
        path_to_mnist_dataset,
        labels_to_include=[0, 1],
        n_samples_per_label=10,
        resize=(16, 16),
    )

    circuits_directory = "C:\\test_circuits"

    print("Running")
    # compute_and_save_circuits(X, y, save_dir=circuits_directory, number_of_workers=4)
    print("Circuits computed")
    train_QCNN(epochs=10, num_classes=2, circuits_directory=circuits_directory)