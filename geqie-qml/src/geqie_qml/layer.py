import os
import numpy as np
import torch
import torch.nn as nn

from contextlib import contextmanager
from concurrent import futures
from multiprocessing import cpu_count
from torch.utils.data import Dataset
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import UnitaryGate
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.gradients import SPSASamplerGradient
from qiskit_machine_learning.neural_networks import SamplerQNN

from qiskit.qasm2 import dumps as qasm2_dumps, loads as qasm2_loads

from geqie_qml.qec import QECCode
from geqie_qml.noise_suppression.twirling import twirl_circuit
from geqie_qml.noise_suppression.dynamical_decoupling import (
    DD_SEQUENCES,
    apply_dd,
)
from geqie_qml.ansatze import default_vqc_ansatz


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


def _identity_interpret(x):
    """
    Identity interpret function for SamplerQNN.

    SamplerQNN requires both ``interpret`` and ``output_shape`` to be set
    together when overriding the default ``2**num_qubits`` output size.
    Defined at module level so ProcessPoolExecutor workers can pickle it.
    """
    return x


def _build_qec_full_circuit(matrix_np, num_logical_qubits, num_layers,
                            qec_encoding_qasm=None, num_physical_qubits=None,
                            measure_logical_only=False, twirl_seed=None,
                            dd_sequence_name=None, backend_target=None,
                            ansatz_factory=None, output_qubits=None):
    """
    Build the full quantum circuit for a single sample, optionally with QEC.

    When *qec_encoding_qasm* is None the circuit is the standard
    ``[ImageUnitary] → [VQC]`` on *num_logical_qubits*.

    When provided, the circuit becomes::

        |0⟩^N → [ImageUnitary on logical qubits] → [QEC encode] → [VQC on N qubits]

    where N = *num_physical_qubits*.

    The ansatz itself is produced by *ansatz_factory* (default:
    ``default_vqc_ansatz``).  The factory is invoked as
    ``ansatz_factory(n_total, num_layers=num_layers, output_qubits=output_qubits)``;
    factories that don't take one of the kwargs can absorb it with ``**_``.

    Measurement
    -----------
    If *measure_logical_only* is True (only meaningful when QEC is active),
    a ``ClassicalRegister`` of width ``num_logical_qubits`` is added and
    only the first ``num_logical_qubits`` qubits are measured.  The
    resulting ``SamplerQNN`` will then produce a probability vector of
    length ``2 ** num_logical_qubits``.

    Otherwise no explicit measurement is added and ``SamplerQNN`` will
    measure all qubits, producing a vector of length ``2 ** n_total`` —
    unless the ansatz itself adds measurements (e.g. ``QCNN_layer``), in
    which case its classical register width determines the output.

    Pauli twirling
    --------------
    When *twirl_seed* is not None the VQC ansatz is rewritten with random
    Pauli twirls applied around every two-qubit entangling gate, using
    ``numpy.random.default_rng(twirl_seed)`` so the rewrite is deterministic
    given the seed.  Only the VQC is twirled; the image unitary and the
    QEC encoder are composed in untouched.

    Dynamical decoupling
    --------------------
    When *dd_sequence_name* is not None the FULL composed circuit
    (image unitary + QEC encoder + VQC, possibly already twirled) is
    ALAP-scheduled against *backend_target* and its idle windows are
    padded with the named DD pulse train.  DD runs last so the inserted
    pulses see the final, ready-to-execute circuit.
    """
    if ansatz_factory is None:
        ansatz_factory = default_vqc_ansatz

    if qec_encoding_qasm is not None:
        n_total = num_physical_qubits
        vqc = ansatz_factory(n_total, num_layers=num_layers, output_qubits=output_qubits)
        if twirl_seed is not None:
            vqc = twirl_circuit(vqc, np.random.default_rng(twirl_seed))
        qc = QuantumCircuit(n_total)
        # Image unitary acts only on the first num_logical_qubits
        qc.append(UnitaryGate(matrix_np), range(num_logical_qubits))
        # QEC encoding circuit (reconstructed from QASM to stay picklable)
        qec_qc = qasm2_loads(qec_encoding_qasm)
        qc.compose(qec_qc, inplace=True)
        # VQC on all physical qubits
        qc.compose(vqc, inplace=True)

        if measure_logical_only:
            creg = ClassicalRegister(num_logical_qubits, name="c_logical")
            qc.add_register(creg)
            for i in range(num_logical_qubits):
                qc.measure(i, creg[i])
    else:
        n_total = num_logical_qubits
        vqc = ansatz_factory(n_total, num_layers=num_layers, output_qubits=output_qubits)
        if twirl_seed is not None:
            vqc = twirl_circuit(vqc, np.random.default_rng(twirl_seed))
        qc = QuantumCircuit(n_total)
        qc.append(UnitaryGate(matrix_np), range(n_total))
        qc.compose(vqc, inplace=True)

    if dd_sequence_name is not None:
        qc = apply_dd(qc, backend_target, sequence_name=dd_sequence_name)
    return qc


def _worker_forward_eval(args):
    """
    Forward pass for a single sample.  Called inside a worker process.

    Returns
    -------
    np.ndarray, shape (output_size,)
        Probability distribution over basis states.  ``output_size`` is
        ``2 ** num_logical_qubits`` when *measure_logical_only* is True,
        otherwise ``2 ** num_total_qubits``.
    """
    (matrix_np, weights_np, num_qubits, num_layers, shots,
     qec_encoding_qasm, num_physical_qubits, measure_logical_only,
     twirl_seed, dd_sequence_name, backend_target,
     ansatz_factory, output_qubits) = args
    qc = _build_qec_full_circuit(
        matrix_np, num_qubits, num_layers,
        qec_encoding_qasm, num_physical_qubits,
        measure_logical_only=measure_logical_only,
        twirl_seed=twirl_seed,
        dd_sequence_name=dd_sequence_name,
        backend_target=backend_target,
        ansatz_factory=ansatz_factory,
        output_qubits=output_qubits,
    )

    sampler = Sampler(default_shots=shots)
    qnn_kwargs = dict(
        circuit=qc,
        input_params=None,
        weight_params=list(qc.parameters),
        sampler=sampler,
        gradient=SPSASamplerGradient(sampler=sampler),
    )
    if measure_logical_only:
        # Force SamplerQNN to size the output by the *measured* (logical)
        # qubits rather than the full physical register.  Without this it
        # would default to 2**num_qubits_in_circuit (= physical) regardless
        # of the actual classical register width.
        qnn_kwargs["output_shape"] = 2 ** num_qubits
        qnn_kwargs["interpret"] = _identity_interpret
    qnn = SamplerQNN(**qnn_kwargs)
    return qnn.forward(input_data=None, weights=weights_np).flatten()


def _worker_grad_eval(args):
    """
    SPSA gradient for a single sample.  Called inside a worker process.

    Returns
    -------
    np.ndarray, shape (output_size, num_weights)
        Jacobian of the output probabilities w.r.t. the quantum weights.
    """
    (matrix_np, weights_np, num_qubits, num_layers, shots,
     qec_encoding_qasm, num_physical_qubits, measure_logical_only,
     twirl_seed, dd_sequence_name, backend_target,
     ansatz_factory, output_qubits) = args
    qc = _build_qec_full_circuit(
        matrix_np, num_qubits, num_layers,
        qec_encoding_qasm, num_physical_qubits,
        measure_logical_only=measure_logical_only,
        twirl_seed=twirl_seed,
        dd_sequence_name=dd_sequence_name,
        backend_target=backend_target,
        ansatz_factory=ansatz_factory,
        output_qubits=output_qubits,
    )

    sampler = Sampler(default_shots=shots)
    grad_fn = SPSASamplerGradient(sampler=sampler)
    qnn_kwargs = dict(
        circuit=qc,
        input_params=None,
        weight_params=list(qc.parameters),
        sampler=sampler,
        gradient=grad_fn,
    )
    if measure_logical_only:
        qnn_kwargs["output_shape"] = 2 ** num_qubits
        qnn_kwargs["interpret"] = _identity_interpret
    qnn = SamplerQNN(**qnn_kwargs)
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
    weights              : (num_params,) tensor — the trainable quantum weights
    executor             : ProcessPoolExecutor | None
    matrices_np_list     : list of np.ndarray, each (2^n, 2^n) image unitary
    num_qubits           : int   — logical qubit count (image unitary size)
    num_layers           : int
    shots                : int
    qec_encoding_qasm    : str | None — OpenQASM string of the QEC encoding circuit
    num_physical_qubits  : int | None — total physical qubit count (with QEC)
    measure_logical_only : bool — when True, measure only the first
                           ``num_qubits`` (logical) qubits, yielding a
                           ``2**num_qubits``-dim probability vector.
    ansatz_factory       : callable — builds the VQC ansatz; signature
                           ``(n_qubits, num_layers=..., output_qubits=...)``.
    output_qubits        : int | None — forwarded to the ansatz factory.
    """

    @staticmethod
    def forward(ctx, weights, executor, matrices_np_list,
                num_qubits, num_layers, shots,
                qec_encoding_qasm, num_physical_qubits,
                measure_logical_only, twirl_seeds,
                dd_sequence_name, backend_target,
                ansatz_factory, output_qubits):
        weights_np = weights.detach().numpy()

        # twirl_seeds is a list of length len(matrices_np_list); each entry
        # is either an int (twirl with that seed) or None (no twirling).
        # Backward must use the SAME seeds so SPSA evaluates the same
        # twirled circuit at the perturbed parameter points.
        args_list = [
            (matrix, weights_np, num_qubits, num_layers, shots,
             qec_encoding_qasm, num_physical_qubits, measure_logical_only,
             twirl_seed, dd_sequence_name, backend_target,
             ansatz_factory, output_qubits)
            for matrix, twirl_seed in zip(matrices_np_list, twirl_seeds)
        ]
        probs_list = _work_dispatch(_worker_forward_eval, args_list, executor)
        probs_np = np.stack(probs_list)  # (batch_size, output_size)

        ctx.save_for_backward(weights)
        ctx.executor = executor
        ctx.matrices_np_list = matrices_np_list
        ctx.num_qubits = num_qubits
        ctx.num_layers = num_layers
        ctx.shots = shots
        ctx.qec_encoding_qasm = qec_encoding_qasm
        ctx.num_physical_qubits = num_physical_qubits
        ctx.measure_logical_only = measure_logical_only
        ctx.twirl_seeds = twirl_seeds
        ctx.dd_sequence_name = dd_sequence_name
        ctx.backend_target = backend_target
        ctx.ansatz_factory = ansatz_factory
        ctx.output_qubits = output_qubits

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
            (m, weights_np, ctx.num_qubits, ctx.num_layers, ctx.shots,
             ctx.qec_encoding_qasm, ctx.num_physical_qubits,
             ctx.measure_logical_only, twirl_seed,
             ctx.dd_sequence_name, ctx.backend_target,
             ctx.ansatz_factory, ctx.output_qubits)
            for m, twirl_seed in zip(ctx.matrices_np_list, ctx.twirl_seeds)
        ]
        jac_list = _work_dispatch(_worker_grad_eval, args_list, ctx.executor)

        jac_np = np.stack(jac_list)             # (batch, output, num_weights)
        grad_np = grad_output.detach().numpy()  # (batch, output)

        # Chain rule: accumulate over batch and output dimensions
        # d_loss/d_w = sum_{b,o} (d_loss/d_prob_{b,o}) * (d_prob_{b,o}/d_w)
        weight_grad_np = np.einsum("bo,bop->p", grad_np, jac_np)

        return (
            torch.tensor(weight_grad_np, dtype=weights.dtype),  # weights
            None,   # executor
            None,   # matrices_np_list
            None,   # num_qubits
            None,   # num_layers
            None,   # shots
            None,   # qec_encoding_qasm
            None,   # num_physical_qubits
            None,   # measure_logical_only
            None,   # twirl_seeds
            None,   # dd_sequence_name
            None,   # backend_target
            None,   # ansatz_factory
            None,   # output_qubits
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
    qec_code : QECCode | None
        Optional QEC encoding to apply between the image unitary and the
        VQC ansatz.  When provided, the VQC operates on the expanded
        physical-qubit register.
    measure_logical_only : bool
        Only meaningful when ``qec_code`` is provided.  When True (default),
        measure only the first ``num_qubits`` (logical) qubits, so the
        output dimension stays ``2 ** num_qubits`` regardless of the QEC
        ancilla overhead.  When False, all physical qubits are measured
        and the output dimension grows to ``2 ** num_physical_qubits``.
    pauli_twirling : bool
        When True, every two-qubit entangling gate in the VQC ansatz is
        wrapped in a random Pauli sandwich on every forward pass.  In the
        noiseless limit this leaves the circuit invariant; under coherent
        two-qubit gate noise, averaging over training shots projects the
        noise onto a stochastic Pauli channel.  Trainable parameters and
        gradient computation are unaffected — twirling is purely a
        circuit rewrite at execution time.
    twirl_seed : int | None
        Seed for the twirling rewrite.  Combined with an internal per-call
        counter so consecutive forward passes get fresh random twirls
        while the run remains reproducible given ``twirl_seed``.  Pass
        ``None`` to draw OS entropy on every call.
    dynamical_decoupling : str | None
        Name of a DD pulse train (one of ``"XX"``, ``"XY4"``, ``"KDD"``)
        to apply just before execution.  ``None`` (default) disables DD.
        DD runs after Pauli twirling and before the circuit reaches the
        sampler, so the inserted pulses see the final, ready-to-execute
        circuit (image unitary + QEC + ansatz, possibly twirled).
    backend_target : qiskit.transpiler.Target | None
        Required when ``dynamical_decoupling`` is set: provides the gate
        durations the ALAP scheduler and DD padder need to compute idle
        windows.  Use ``backend.target`` of a fake or real IBM backend.
        Note that ``StatevectorSampler`` is duration-less, so DD inserts
        pulses but the simulator will treat them as plain gates — DD only
        produces meaningful behaviour against a noise-aware backend.
    ansatz_factory : callable | None
        Factory that builds the parameterised VQC ansatz, invoked as
        ``ansatz_factory(n_qubits, num_layers=num_layers, output_qubits=output_qubits)``.
        Defaults to ``default_vqc_ansatz`` (brickwork Rx/Ry/Rz + CX).  Pass
        ``QCNN_layer`` (or any callable with the same signature, using
        ``**_`` to absorb unused kwargs) to swap in a different ansatz.
        When QEC is active the factory is sized to the physical register.
    output_qubits : int | None
        Forwarded to the ansatz factory.  Used by ``QCNN_layer`` to set the
        width of its classical register and the final measurement; ignored
        by ``default_vqc_ansatz``.

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
        qec_code: QECCode | None = None,
        measure_logical_only: bool = True,
        pauli_twirling: bool = False,
        twirl_seed: int | None = None,
        dynamical_decoupling: str | None = None,
        backend_target=None,
        ansatz_factory=None,
        output_qubits: int | None = None,
    ):
        super().__init__()
        self.num_qubits = num_qubits  # logical qubits (image unitary size)
        self.num_layers = num_layers
        self.num_shots = shots
        self.scale_output = scale_output
        self.qec_code = qec_code
        # measure_logical_only is only meaningful when QEC is active; without
        # QEC there are no ancillas, so logical == total and the flag is a no-op.
        self.measure_logical_only = measure_logical_only and (qec_code is not None)

        # Pauli-twirling rewrites every two-qubit gate in the VQC ansatz with
        # random Pauli sandwiches per forward pass.  The trainable parameter
        # tensors are unchanged — twirling is a circuit rewrite at execution
        # time only.  twirl_seed seeds the rewrite; the per-call counter
        # advances every forward pass so consecutive passes get fresh random
        # twirls while the whole run remains reproducible given twirl_seed.
        self.pauli_twirling = pauli_twirling
        self.twirl_seed = twirl_seed
        self._twirl_call_counter = 0

        # Dynamical decoupling — purely an execution-time circuit rewrite.
        # Validated up front so misconfiguration surfaces at construction
        # rather than deep inside a worker process during forward().
        if dynamical_decoupling is not None:
            if dynamical_decoupling not in DD_SEQUENCES:
                raise ValueError(
                    f"Unknown dynamical_decoupling sequence "
                    f"{dynamical_decoupling!r}; supported: {sorted(DD_SEQUENCES)}."
                )
            if backend_target is None:
                raise ValueError(
                    "dynamical_decoupling requires per-gate duration information. "
                    "Pass `backend.target` from a fake or real IBM backend "
                    "(e.g. FakeKyoto, FakeBrisbane, ibm_brisbane). Duration-less "
                    "simulators such as StatevectorSampler cannot supply gate "
                    "lengths, so the scheduler has no idle windows to fill."
                )
        self.dynamical_decoupling = dynamical_decoupling
        self.backend_target = backend_target

        # When QEC is active, the VQC operates on the expanded physical-qubit
        # register.  The image unitary still acts only on num_qubits (logical).
        if qec_code is not None:
            self._num_total_qubits = qec_code.num_physical_qubits(num_qubits)
            qec_qc = qec_code.encoding_circuit(num_qubits)
            self._qec_encoding_qasm: str | None = qasm2_dumps(qec_qc)
        else:
            self._num_total_qubits = num_qubits
            self._qec_encoding_qasm = None

        # Custom ansatz support.  Workers reconstruct the circuit
        # independently in each forward call, so the factory itself is what
        # must be picklable (passed through ProcessPoolExecutor).
        self.ansatz_factory = ansatz_factory or default_vqc_ansatz
        self.output_qubits = output_qubits

        # Trainable quantum weights, registered as a proper nn.Parameter so
        # that optimisers, state_dict, and requires_grad all work out of the box.
        # Build a probe circuit to size the parameter vector — needed because
        # ansatze like QCNN_layer add parameters dynamically as the circuit
        # is constructed, so 3 * n * L is not always the right count.
        probe_circuit = self.ansatz_factory(
            self._num_total_qubits,
            num_layers=num_layers,
            output_qubits=output_qubits,
        )
        num_params = len(probe_circuit.parameters)
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

    def _draw_twirl_seeds(self, batch_size: int) -> list[int | None]:
        """
        Generate one twirl seed per sample for the current forward pass.

        Returns a list of ``None`` when ``pauli_twirling`` is disabled, so
        the worker path skips the rewrite altogether.  Otherwise mixes
        ``twirl_seed`` with the per-instance call counter via
        ``numpy.random.SeedSequence`` to derive a deterministic but
        forward-pass-specific seed sequence, then spawns one child seed
        per sample.  The counter advances even when the run is unseeded
        (``twirl_seed is None``), in which case ``SeedSequence`` pulls
        fresh OS entropy on every call.
        """
        if not self.pauli_twirling:
            return [None] * batch_size

        if self.twirl_seed is None:
            ss = np.random.SeedSequence()
        else:
            ss = np.random.SeedSequence([int(self.twirl_seed), self._twirl_call_counter])
        self._twirl_call_counter += 1

        sample_ss = ss.spawn(batch_size)
        return [int(s.generate_state(1, dtype=np.uint32)[0]) for s in sample_ss]

    @property
    def output_dim(self) -> int:
        """
        Dimension of the output probability vector.

        - Without QEC: ``2 ** num_qubits``
        - With QEC and ``measure_logical_only=True``: ``2 ** num_qubits``
          (only the logical qubits are measured, ancillas are traced out)
        - With QEC and ``measure_logical_only=False``: ``2 ** num_total_qubits``
          (full physical-register measurement)
        """
        if self.measure_logical_only:
            return 2 ** self.num_qubits
        return 2 ** self._num_total_qubits

    def forward(self, batched_matrices: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        batched_matrices : torch.Tensor, shape (batch_size, 2**n, 2**n), complex
            Batch of pre-encoded image unitary matrices.  Here *n* is
            ``num_qubits`` (the logical qubit count).

        Returns
        -------
        torch.Tensor, shape (batch_size, output_dim), float
            Probability distribution over the measured basis states for
            each sample.  See ``output_dim`` for the exact size.
            If ``scale_output=True``, values are multiplied by ``output_dim``.
        """
        logical_dim = 2 ** self.num_qubits
        if batched_matrices.shape[1:] != torch.Size([logical_dim, logical_dim]):
            raise ValueError(
                f"VQCLayer expects unitary matrices of shape {(logical_dim, logical_dim)}, "
                f"but got input with shape {tuple(batched_matrices.shape[1:])}. "
                f"Check that num_qubits={self.num_qubits} matches your encoded data."
            )

        matrices_np_list = [m.detach().cpu().numpy() for m in batched_matrices]
        twirl_seeds = self._draw_twirl_seeds(len(matrices_np_list))

        # QNNBatchFunction runs sequentially when self._executor is None,
        # or fans out to the process pool when parallel_context() is active.
        probs = QNNBatchFunction.apply(
            self.quantum_weight,
            self._executor,
            matrices_np_list,
            self.num_qubits,
            self.num_layers,
            self.num_shots,
            self._qec_encoding_qasm,
            self._num_total_qubits if self.qec_code is not None else None,
            self.measure_logical_only,
            twirl_seeds,
            self.dynamical_decoupling,
            self.backend_target,
            self.ansatz_factory,
            self.output_qubits,
        )

        if self.scale_output:
            probs = probs * self.output_dim

        return probs

    def extra_repr(self) -> str:
        """Adds layer details to the standard nn.Module string representation."""
        parts = [
            f"num_qubits={self.num_qubits}",
            f"num_layers={self.num_layers}",
            f"shots={self.num_shots}",
            f"scale_output={self.scale_output}",
            f"num_params={self.quantum_weight.numel()}",
        ]
        if self.ansatz_factory is not default_vqc_ansatz:
            parts.append(f"ansatz={self.ansatz_factory.__name__}")
        if self.output_qubits is not None:
            parts.append(f"output_qubits={self.output_qubits}")
        if self.qec_code is not None:
            parts.append(f"qec={self.qec_code.name}")
            parts.append(f"total_qubits={self._num_total_qubits}")
            parts.append(f"measure_logical_only={self.measure_logical_only}")
            parts.append(f"output_dim={self.output_dim}")
        if self.pauli_twirling:
            parts.append(f"pauli_twirling=True")
            parts.append(f"twirl_seed={self.twirl_seed}")
        if self.dynamical_decoupling is not None:
            parts.append(f"dynamical_decoupling={self.dynamical_decoupling!r}")
        return ", ".join(parts)
