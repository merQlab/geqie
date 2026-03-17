import time
import os
import glob
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from concurrent import futures
from multiprocessing import cpu_count
from datasets import load_dataset
from torch.nn import Linear, CrossEntropyLoss, MSELoss, NLLLoss
from torch.optim import LBFGS, Adam
from torch.utils.data import Dataset, DataLoader
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Operator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.gradients import SPSASamplerGradient

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

        jac_np = np.stack(jac_list)         # (batch, output, num_weights)
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
# QNN Module
# ---------------------------------------------------------------------------

class QNN_Pythorch_Module(nn.Module):
    def __init__(self, num_classes, num_qubits=8, num_layers=3, shots=1024):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_classes = num_classes
        self.num_shots = shots
        self.num_layers = num_layers  # stored so workers can reconstruct

        # 1. Create the parameterized VQC circuit ONCE
        self.vqc_circuit = QuantumCircuit(num_qubits)
        self.thetas = ParameterVector("theta", length=3 * num_qubits * num_layers)

        # Apply the brickwork architecture to the VQC circuit
        for layer in range(num_layers):
            offset = layer * 3 * num_qubits
            for i in range(num_qubits):
                self.vqc_circuit.rx(self.thetas[offset + i], i)
            for i in range(num_qubits):
                self.vqc_circuit.ry(self.thetas[offset + num_qubits + i], i)
            for i in range(num_qubits):
                self.vqc_circuit.rz(self.thetas[offset + 2*num_qubits + i], i)

            # Alternating brickwork entanglement
            if layer % 2 == 0:
                for i in range(0, num_qubits - 1, 2):
                    self.vqc_circuit.cx(i, i + 1)
            else:
                for i in range(1, num_qubits - 1, 2):
                    self.vqc_circuit.cx(i, i + 1)
                self.vqc_circuit.cx(num_qubits - 1, 0)

        # 2. Register the quantum weights directly as a native PyTorch Parameter.
        self.quantum_weight = nn.Parameter(
            torch.empty(len(self.thetas)).uniform_(-np.pi, np.pi)
        )

        # 3. Classical head
        self.classical_head = nn.Linear(2**num_qubits, num_classes)

        # 4. Optional executor injected by the training loop (see train_QCNN).
        #    When set, forward() uses the fast parallel path.
        #    When None, falls back to the original sequential path.
        self.executor = None

    # ------------------------------------------------------------------
    # Sequential forward (original, kept as fallback / reference)
    # ------------------------------------------------------------------
    # def _forward_sequential(self, batched_matrices):
    #     batched_probs = []
    #     for single_matrix_tensor in batched_matrices:
    #         matrix_np = single_matrix_tensor.detach().cpu().numpy()
    #         qc = QuantumCircuit(self.num_qubits)
    #         qc.append(UnitaryGate(matrix_np), range(self.num_qubits))
    #         qc.compose(self.vqc_circuit, inplace=True)

    #         fresh_sampler = Sampler(default_shots=self.num_shots)
    #         fresh_qnn = SamplerQNN(
    #             circuit=qc,
    #             input_params=None,
    #             weight_params=list(qc.parameters),
    #             sampler=fresh_sampler,
    #             gradient=ParamShiftSamplerGradient(sampler=fresh_sampler)
    #         )
    #         probs = _TorchNNFunction.apply(
    #             torch.empty(0),
    #             self.quantum_weight,
    #             fresh_qnn,
    #             False
    #         )
    #         batched_probs.append(probs)

    #     return torch.stack(batched_probs)

    # ------------------------------------------------------------------
    # Parallel forward — uses ParallelQNNBatchFunction
    # ------------------------------------------------------------------
    def _forward_parallel(self, batched_matrices):
        matrices_np_list = [
            m.detach().cpu().numpy() for m in batched_matrices
        ]
        # ParallelQNNBatchFunction.apply fans out workers for both the
        # forward simulation and the parameter-shift backward pass.
        return ParallelQNNBatchFunction.apply(
            self.quantum_weight,
            self.executor,
            matrices_np_list,
            self.num_qubits,
            self.num_layers,
            self.num_shots,
        )

    # ------------------------------------------------------------------
    # Main forward — dispatches to parallel or sequential path
    # ------------------------------------------------------------------
    def forward(self, batched_matrices):
        # batched_matrices: (batch_size, 2^n, 2^n) complex tensor
        if self.executor is not None:
            batched_probs = self._forward_parallel(batched_matrices)
        else:
            batched_probs = self._forward_sequential(batched_matrices)

        # Feature scaling
        scaled_probs = batched_probs * (2 ** self.num_qubits)

        logits = self.classical_head(scaled_probs)
        return torch.log_softmax(logits, dim=-1)

# ---------------------------------------------------------------------------
# Training loop — now with parallel circuit evaluation
# ---------------------------------------------------------------------------

def train_QCNN(epochs=20, num_classes=2, batch_size=5,
               circuits_directory="circuits", num_workers=None):
    """
    Train the QNN model.

    Parameters
    ----------
    num_workers : int | None
        Number of worker processes.  Defaults to (cpu_count - 1), min 1.
        Each worker handles one sample's circuit simulation independently,
        so setting this to the number of physical cores works well.
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    f_loss = NLLLoss()
    qnn_model = QNN_Pythorch_Module(
        num_classes=num_classes, num_qubits=9, num_layers=1
    )

    optimizer = Adam([
        {'params': qnn_model.quantum_weight, 'lr': 0.001},
        {'params': qnn_model.classical_head.parameters(), 'lr': 0.01}
    ])

    qnn_model.train()
    loss_list = []

    files = sorted(glob.glob(f"{circuits_directory}/*.npz"))
    print(len(files))
    dataset = MatrixDataset(files)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"Starting training with {num_workers} worker processes")

    # Create the pool ONCE outside the epoch loop.
    # Workers are initialised with thread-count limits to prevent
    # oversubscription (each worker is single-threaded; N workers → N cores).
    with futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_init_training_worker,
        # "spawn" avoids fork-safety issues with Qiskit's internal threads.
        mp_context=__import__("multiprocessing").get_context("spawn"),
    ) as executor:

        # Inject the live executor into the model so forward() can use it.
        qnn_model.executor = executor

        for epoch in range(epochs):
            start = time.time()
            total_loss = []

            for batch_matrices, batch_labels in dataloader:
                optimizer.zero_grad()

                # forward() now fans out all batch_size circuit simulations
                # in parallel, then chains gradients via ParallelQNNBatchFunction
                output = qnn_model(batch_matrices)

                loss = f_loss(output, batch_labels)
                loss.backward()   # triggers parallel parameter-shift evals

                total_loss.append(loss.item())
                optimizer.step()

            loss_list.append(sum(total_loss) / len(total_loss))
            elapsed = time.time() - start
            print(f"Epoch {epoch + 1}/{epochs}  "
                  f"Loss: {loss_list[-1]:.4f}  "
                  f"Time: {elapsed:.2f}s")

        # Remove the executor reference before it closes
        qnn_model.executor = None

    return loss_list, qnn_model


# ---------------------------------------------------------------------------
# Original batch training function (kept for comparison)
# ---------------------------------------------------------------------------

def train_QCNN_batch_old(epochs=20, num_classes=2, circuits_directory="circuits"):
    f_loss = NLLLoss()
    qnn_model = QNN_Pythorch_Module(num_classes=num_classes, num_qubits=9, num_layers=1)

    optimizer = Adam([
        {'params': qnn_model.quantum_weight, 'lr': 0.001},
        {'params': qnn_model.classical_head.parameters(), 'lr': 0.01}
    ])

    qnn_model.train()
    loss_list = []
    batch_files = sorted(glob.glob(f"{circuits_directory}/batch_*.npz"))

    for epoch in range(epochs):
        start = time.time()
        total_loss = []

        for batch_file in batch_files:
            data = np.load(batch_file)
            matrices = data["matrices"]
            labels = data["labels"]

            optimizer.zero_grad()

            for matrix, label in zip(matrices, labels):
                output = qnn_model(matrix)
                label_tensor = torch.tensor(label, dtype=torch.long)
                loss = f_loss(output, label_tensor) / len(matrices)
                loss.backward()
                total_loss.append(loss.item() * len(matrices))

            optimizer.step()

        loss_list.append(sum(total_loss) / len(total_loss))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss_list[-1]:.4f}")
        end = time.time()
        print(f"Time: {end - start:.2f}s")

    return loss_list, qnn_model


# ---------------------------------------------------------------------------
# Data loading & circuit precomputation  (unchanged)
# ---------------------------------------------------------------------------

def _load_and_process_mnist_dataset(path_to_mnist_dataset, labels_to_include=[0, 1],
                                   n_samples_per_label=100, resize=(8, 8)):
    mnist_dataset = load_dataset("parquet", data_files=path_to_mnist_dataset)

    selected_images = []
    for image_idx in range(len(mnist_dataset["train"])):
        if mnist_dataset["train"][image_idx]["label"] in labels_to_include:
            img_8x8 = mnist_dataset["train"][image_idx]["image"].resize(
                resize, resample=Image.BILINEAR
            )
            img = np.array(img_8x8)
            selected_images.append({
                "image": img,
                "label": mnist_dataset["train"][image_idx]["label"]
            })

    X = np.array([item["image"] for item in selected_images])
    y = np.array([item["label"] for item in selected_images])
    return X, y

def load_and_process_mnist_dataset(path_to_mnist_dataset, labels_to_include=[0, 1],
                                   n_samples_per_label=100, resize=(8, 8)):
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


def precompute_and_save_circuits_batched(X_data, y_data, batch_size=5, save_dir="circuits"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_samples = len(X_data)

    for i in range(0, num_samples, batch_size):
        batch_X = X_data[i : i + batch_size]
        batch_y = y_data[i : i + batch_size]
        batch_matrices = []

        for img in batch_X:
            qc = geqie.encode(frqi.init_function, frqi.data_function,
                              frqi.map_function, img)
            flat_qc = qc.decompose()
            while len(flat_qc.data) != len(flat_qc.decompose().data):
                flat_qc = flat_qc.decompose()

            pure_qc = QuantumCircuit(flat_qc.num_qubits)
            for instruction in flat_qc.data:
                op_name = (instruction.operation.name
                           if hasattr(instruction, 'operation')
                           else instruction[0].name)
                if op_name not in ['reset', 'measure', 'barrier']:
                    pure_qc.append(instruction)

            unitary_matrix = np.array(Operator(pure_qc).data, dtype=np.complex128)
            batch_matrices.append(unitary_matrix)

        batch_filename = os.path.join(save_dir, f"batch_{i//batch_size}.npz")
        np.savez(batch_filename, matrices=batch_matrices, labels=batch_y)
        print(f"Saved {batch_filename} with dtype {batch_matrices[0].dtype}")


def precompute_and_save_circuits_short(X_data, y_data,
                                       samples_numerals: list[int] = None,
                                       save_dir="circuits"):
    print(samples_numerals)
    os.makedirs(save_dir, exist_ok=True)

    if samples_numerals is None:
        samples_numerals = range(len(X_data))

    for input_matrix, label, sample_number in zip(X_data, y_data, samples_numerals):
        qc = geqie.encode(frqi.init_function, frqi.data_function,
                          frqi.map_function, input_matrix)
        for instr in qc.data:
            if isinstance(instr.operation, UnitaryGate):
                unitary = instr.operation.to_matrix()
                break

        filename = os.path.join(save_dir, f"matrix_{sample_number}")
        np.savez(filename, matrix=unitary, label=label, dtype=np.complex128)

_U_INIT_CACHE = {}

def _compute_circuit(image, geqie_encoding=frqi):
    global _sim
    qc = geqie.encode(geqie_encoding.init_function, geqie_encoding.data_function,
                      geqie_encoding.map_function, image, perform_measurement=False)
    qc_d = qc.decompose()
    n_qubits = qc_d.num_qubits

    rest_qc = QuantumCircuit(n_qubits)
    state_prep_qc = QuantumCircuit(n_qubits)

    for instr in qc_d.data:
        if instr.operation.name == 'state_preparation':
            state_prep_qc.append(instr.operation, instr.qubits)
        if instr.operation.name not in ['reset', 'measure', 'barrier', 'state_preparation']:
            rest_qc.append(instr.operation, instr.qubits)

    rest_qc.save_unitary()
    U_rest = np.array(_sim.run(rest_qc).result().data()['unitary'])

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
    _sim = AerSimulator(method='unitary', max_parallel_threads=1,
                        max_parallel_shots=1, max_parallel_experiments=1)

def _compute_save_single(args):
    """Single-sample worker — picklable at module level."""
    image, label, sample_number, save_dir, file_prefix = args
    filename = os.path.join(save_dir, f"{file_prefix}_{sample_number}")
    unitary_matrix = _compute_circuit(image)
    np.savez(filename, matrix=unitary_matrix, label=label, dtype=np.complex128)
    print(f"{filename} saved")

def compute_and_save_circuits(data, labels, save_dir="new_pool_circuits",
                              file_prefix="matrix", number_of_workers=None):
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
        labels_to_include=[x for x in range(10)],
        n_samples_per_label=100,
        resize=(16, 16)
    )

    circuits_directory = 'circuits'

    print("Running")
    compute_and_save_circuits(X, y, save_dir=circuits_directory, 
                              number_of_workers=4)
    print("Circuits computed")
    train_QCNN(epochs= 50, num_classes=10 ,circuits_directory=circuits_directory)
