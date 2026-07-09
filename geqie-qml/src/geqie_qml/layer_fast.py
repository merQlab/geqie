import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression


# ---------------------------------------------------------------------------
# Public nn.Modules
# ---------------------------------------------------------------------------

class FastGEQIELayer(nn.Module):
    """
    Input layer — fast equivalent of :class:`~geqie_qml.layer_v2.GEQIELayer`.

    Converts a pre-computed image unitary ``U`` into the encoded statevector
    ``U|0⟩`` by extracting the first column.  This is the only operation that
    ``GEQIELayer`` performs on the zero state, but without the overhead of
    Qiskit's parameterized circuit machinery.

    No trainable parameters.  Output is a complex statevector ready to be
    passed to :class:`FastAnsatzLayer` or any other quantum layer.

    Parameters
    ----------
    num_qubits : int

    Input / Output
    --------------
    Input  : ``(B, dim, dim)`` complex — pre-computed unitary matrices from the
             data-loader (``dim = 2**num_qubits``).
    Output : ``(B, dim)`` complex — encoded statevectors ``U|0⟩``.
    """

    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[-2] == self.dim and x.shape[-1] == self.dim:
            return x[:, :, 0].contiguous()     # U|0⟩ = first column of U
        if x.dim() == 2 and x.shape[-1] == self.dim:
            return x                            # already a statevector
        raise ValueError(
            f"Expected (B, {self.dim}, {self.dim}) or (B, {self.dim}), "
            f"got {tuple(x.shape)}."
        )

    def extra_repr(self) -> str:
        return f"num_qubits={self.num_qubits}, dim={self.dim}"


class FastAnsatzLayer(nn.Module):
    """
    Trainable ansatz layer with analytic parameter-shift gradients.

    Accepts any Qiskit ``QuantumCircuit`` as the ansatz.  The circuit is parsed
    once at construction into a plain Python gate list; no Qiskit objects are
    retained.  All forward and backward computations are pure NumPy statevector
    operations, bypassing Qiskit's parameter-binding machinery entirely.

    Gradients are computed with the parameter-shift rule and **prefix-state
    caching**: the statevector before each parametric gate is saved during the
    forward pass so that the backward pass only re-runs the circuit *suffix*,
    saving ≈50 % of compute vs. a full re-evaluation per parameter.  This
    avoids the ``assign_parameters_mapping`` bottleneck that makes backward
    passes prohibitively slow for large image encodings.

    Designed to be placed after :class:`FastGEQIELayer` in an ``nn.Sequential``
    pipeline, mirroring the original ``pqc.compose(geqie_layer).compose(ansatz)``
    pattern with ``SamplerQNN``.

    Parameters
    ----------
    num_qubits : int
    ansatz : qiskit.QuantumCircuit
        Parameterized ansatz.  Any single-qubit Pauli-axis rotation
        (``rx``/``ry``/``rz`` and equivalents) may be parameterized; any fixed
        gate (``cx``, ``cz``, ``swap``, ``h``, …) is accepted.
    weight_init : torch.Tensor | None
        Initial weights of shape ``(ansatz.num_parameters,)``.
        Defaults to ``Uniform(0, 2π)``.
    seed : int | None
        Random seed for default weight initialization.

    Input / Output
    --------------
    Input  : ``(B, dim)`` complex — statevectors (e.g. from :class:`FastGEQIELayer`).
    Output : ``(B, dim)`` float  — measurement probability distributions
             ``|A(θ)|input⟩|²``.

    Example
    -------
    >>> from qiskit.circuit.library import real_amplitudes
    >>> ansatz = real_amplitudes(7, reps=4)
    >>> model = nn.Sequential(
    ...     FastGEQIELayer(7),
    ...     FastAnsatzLayer(7, ansatz),
    ...     nn.Linear(128, 10),
    ...     nn.LogSoftmax(dim=-1),
    ... )
    """

    def __init__(
        self,
        num_qubits: int,
        ansatz: QuantumCircuit,
        weight_init=None,
        seed: int | None = None,
    ):
        super().__init__()
        if ansatz.num_qubits != num_qubits:
            raise ValueError(
                f"ansatz acts on {ansatz.num_qubits} qubits, expected {num_qubits}."
            )
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        self.ops = parse_circuit(ansatz)        # parsed once; no live Qiskit objects

        n_w = ansatz.num_parameters
        if weight_init is None:
            rng = torch.Generator().manual_seed(seed) if seed is not None else None
            weight_init = (2.0 * np.pi * torch.rand(n_w, generator=rng)).float()
        else:
            weight_init = torch.as_tensor(weight_init, dtype=torch.float32).reshape(-1)
            if weight_init.numel() != n_w:
                raise ValueError(
                    f"weight_init has {weight_init.numel()} elements, expected {n_w}."
                )

        self.weights = nn.Parameter(weight_init)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 2 or states.shape[-1] != self.dim:
            raise ValueError(
                f"Expected (B, {self.dim}) statevectors, got {tuple(states.shape)}."
            )
        return ParameterShiftFunction.apply(
            states, self.weights, self.ops, self.num_qubits
        )

    def extra_repr(self) -> str:
        n_p = sum(1 for op in self.ops if op["type"] == "1q_param")
        n_f = len(self.ops) - n_p
        return (
            f"num_qubits={self.num_qubits}, dim={self.dim}, "
            f"weights={len(self.weights)}, param_gates={n_p}, fixed_gates={n_f}"
        )


# ---------------------------------------------------------------------------
# Autograd function — parameter-shift with prefix-state caching
# ---------------------------------------------------------------------------

class ParameterShiftFunction(torch.autograd.Function):
    """
    Forward: apply the parsed ansatz circuit to a batch of input states and
    return the measurement probability distribution ``|A(θ)v|²``.

    During the forward pass the statevector just *before* each parametric gate
    is saved (prefix cache).  The backward pass uses these snapshots so that
    each parameter-shift evaluation re-runs only the circuit *suffix* starting
    after the shifted gate, instead of the full circuit from scratch.

    The gradient w.r.t. ``states`` is always ``None`` — these are
    pre-computed image encodings, not trainable parameters.
    """

    @staticmethod
    def forward(ctx, states, weights, ops, n_qubits):
        states_np  = states.detach().cpu().numpy().astype(np.complex128)
        weights_np = weights.detach().cpu().numpy().astype(np.float64)

        # Run the circuit, saving the state just before every parametric gate
        prefix_states: list = []
        param_positions: list = []      # (op_index, param_idx)

        psi = states_np
        for i, op in enumerate(ops):
            if op["type"] == "1q_param":
                prefix_states.append(psi)   # snapshot BEFORE applying this gate
                param_positions.append((i, op["param_idx"]))
            psi = apply_op(psi, op, weights_np, n_qubits)

        probs = np.abs(psi) ** 2        # (batch, dim) — real-valued

        ctx.prefix_states   = prefix_states
        ctx.weights_np      = weights_np
        ctx.ops             = ops
        ctx.n_qubits        = n_qubits
        ctx.param_positions = param_positions

        return torch.as_tensor(probs, dtype=weights.dtype, device=weights.device)

    @staticmethod
    def backward(ctx, grad_output):
        prefix_states   = ctx.prefix_states
        weights_np      = ctx.weights_np
        ops             = ctx.ops
        n               = ctx.n_qubits
        param_positions = ctx.param_positions
        grad_np = grad_output.detach().cpu().numpy().astype(np.float64)

        weight_grad = np.zeros(weights_np.shape[0], dtype=np.float64)
        shift = np.pi / 2.0

        for k, (gate_i, param_idx) in enumerate(param_positions):
            op     = ops[gate_i]
            prefix = prefix_states[k]           # (batch, dim), state before gate_i
            theta  = float(weights_np[param_idx])

            # Apply the shifted gate to the cached prefix (no prefix re-run)
            s_plus  = apply_1q(prefix, param_1q(theta + shift, op["gen"]), op["qubits"][0], n)
            s_minus = apply_1q(prefix, param_1q(theta - shift, op["gen"]), op["qubits"][0], n)

            # Run only the suffix — gates after gate_i — with original weights
            s_plus  = run_ops(ops, s_plus,  weights_np, n, start=gate_i + 1)
            s_minus = run_ops(ops, s_minus, weights_np, n, start=gate_i + 1)

            p_plus  = np.abs(s_plus)  ** 2     # (batch, dim)
            p_minus = np.abs(s_minus) ** 2

            # Chain rule: dL/dθ_k = Σ_{b,o} (dL/dp_{b,o}) · 0.5·(p⁺ - p⁻)
            weight_grad[param_idx] += float(np.sum(grad_np * 0.5 * (p_plus - p_minus)))

        grad_weights = torch.as_tensor(
            weight_grad.astype(np.float32),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        # Gradients for: states, weights, ops (not a tensor), n_qubits (not a tensor)
        return None, grad_weights, None, None


# ---------------------------------------------------------------------------
# Circuit parser  (called once at construction, never during training)
# ---------------------------------------------------------------------------

def parse_circuit(ansatz: QuantumCircuit) -> list:
    """
    Parse a Qiskit ansatz circuit into a plain Python list of gate records.

    Each record is a dict with a ``'type'`` key:

    * ``'1q_param'`` — single-qubit Pauli-axis rotation with a symbolic parameter:
      keys ``gen`` (2×2 generator ndarray), ``qubits`` (list[int]),
      ``param_idx`` (int)
    * ``'1q_fixed'`` — single-qubit gate with a fixed matrix:
      keys ``mat`` (2×2 ndarray), ``qubits`` (list[int])
    * ``'2q_fixed'`` — any fixed two-qubit gate (CX, CZ, SWAP, …):
      keys ``mat`` (4×4 ndarray), ``qubits`` (list[int])

    Any single-qubit gate with exactly one symbolic parameter is accepted as a
    parameterized rotation, provided its generator is a Hermitian, involutory
    Pauli operator (see :func:`extract_1q_generator`).  Any fixed gate (single-
    or two-qubit) is accepted; its matrix is fetched once via
    ``gate.to_matrix()`` at parse time.

    Raises
    ------
    ValueError
        If the circuit contains a parameterized two-qubit gate or a gate with
        more than two qubits.
    """
    param_order = {p: i for i, p in enumerate(ansatz.parameters)}
    ops: list = []

    for inst in ansatz.data:
        gate = inst.operation
        qubits = [ansatz.qubits.index(q) for q in inst.qubits]
        name = gate.name.lower()

        symbolic = [
            p for p in gate.params
            if isinstance(p, ParameterExpression) and p.parameters
        ]

        if gate.num_qubits == 1 and symbolic:
            if len(symbolic) != 1 or len(symbolic[0].parameters) != 1:
                raise ValueError(
                    f"Parameterized gate '{name}' does not depend on exactly one "
                    "free parameter; only single-parameter rotations are supported."
                )
            p0 = next(iter(symbolic[0].parameters))
            ops.append({
                "type": "1q_param",
                "gen": extract_1q_generator(gate, p0),
                "qubits": qubits,
                "param_idx": param_order[p0],
            })

        elif gate.num_qubits == 1:
            ops.append({
                "type": "1q_fixed",
                "mat": np.asarray(gate.to_matrix(), dtype=np.complex128),
                "qubits": qubits,
            })

        elif gate.num_qubits == 2:
            if symbolic:
                raise ValueError(
                    f"Parameterized two-qubit gate '{name}' is not supported. "
                    "Only single-qubit Pauli-axis rotations may be parameterized."
                )
            ops.append({
                "type": "2q_fixed",
                "mat": np.asarray(gate.to_matrix(), dtype=np.complex128),
                "qubits": qubits,
            })

        else:
            raise ValueError(
                f"Unsupported gate: '{name}' with {gate.num_qubits} qubits "
                f"and {len(gate.params)} parameters."
            )

    return ops


# ---------------------------------------------------------------------------
# Circuit evaluation helpers
# ---------------------------------------------------------------------------

def apply_op(psi: np.ndarray, op: dict, weights: np.ndarray, n: int) -> np.ndarray:
    """Apply a single parsed gate record to the batch of statevectors."""
    t = op["type"]
    if t == "1q_param":
        return apply_1q(psi, param_1q(float(weights[op["param_idx"]]), op["gen"]),
                        op["qubits"][0], n)
    elif t == "1q_fixed":
        return apply_1q(psi, op["mat"], op["qubits"][0], n)
    else:   # 2q_fixed
        return apply_2q(psi, op["mat"], op["qubits"], n)


def run_ops(
    ops: list,
    psi: np.ndarray,
    weights: np.ndarray,
    n: int,
    start: int = 0,
) -> np.ndarray:
    """Apply ``ops[start:]`` sequentially to the batch of statevectors."""
    for op in ops[start:]:
        psi = apply_op(psi, op, weights, n)
    return psi


# ---------------------------------------------------------------------------
# Statevector gate application  (Qiskit little-endian convention)
#
# When (batch, dim) is reshaped to (batch, 2, ..., 2) in C-order, qubit k
# (0 = LSB in Qiskit) occupies axis  n_qubits - k  (batch is axis 0).
# ---------------------------------------------------------------------------

def apply_1q(psi: np.ndarray, mat: np.ndarray, qubit: int, n: int) -> np.ndarray:
    """
    Apply a 2×2 single-qubit gate to one qubit of a batch of statevectors.

    Parameters
    ----------
    psi   : (batch, dim) complex ndarray
    mat   : (2, 2) complex ndarray
    qubit : target qubit index (0 = LSB in Qiskit convention)
    n     : total number of qubits

    Returns
    -------
    (batch, dim) complex ndarray
    """
    batch, dim = psi.shape
    ax = n - qubit                          # qubit axis in (batch, 2,…,2) tensor
    psi = psi.reshape(batch, *([2] * n))
    psi = np.tensordot(mat, psi, axes=([1], [ax]))
    psi = np.moveaxis(psi, 0, ax)
    return psi.reshape(batch, dim)


def apply_2q(psi: np.ndarray, mat: np.ndarray, qubits: list, n: int) -> np.ndarray:
    """
    Apply a 4×4 two-qubit gate to a batch of statevectors.

    ``mat`` must be in Qiskit's little-endian 2-qubit convention:
    ``mat[2*b1+b0, 2*c1+c0]`` where b0/c0 = first-arg bit, b1/c1 = second-arg
    bit — this matches ``gate.to_matrix()`` for standard gates (CX, CZ, SWAP).

    Parameters
    ----------
    psi    : (batch, dim) complex ndarray
    mat    : (4, 4) complex ndarray
    qubits : [first_gate_arg_idx, second_gate_arg_idx]
    n      : total number of qubits

    Returns
    -------
    (batch, dim) complex ndarray
    """
    batch, dim = psi.shape
    q0, q1 = qubits             # q0 = first gate arg (LSB), q1 = second (MSB)
    ax0 = n - q0                # qubit q0 axis in (batch, 2,…,2) tensor
    ax1 = n - q1                # qubit q1 axis
    mat4 = mat.reshape(2, 2, 2, 2)      # (out_q1, out_q0, in_q1, in_q0)
    psi = psi.reshape(batch, *([2] * n))
    psi = np.tensordot(mat4, psi, axes=([2, 3], [ax1, ax0]))
    psi = np.moveaxis(psi, [0, 1], [ax1, ax0])
    return psi.reshape(batch, dim)


# ---------------------------------------------------------------------------
# Generic single-qubit Pauli-rotation gate
#
# Any gate of the form  U(θ) = exp(-i θ P / 2) = cos(θ/2) I − i sin(θ/2) P
# with a Hermitian, involutory generator (P² = I) is supported.  The generator
# P is extracted once from the Qiskit gate at parse time via  P = i·U(π), then
# the matrix at any θ is rebuilt cheaply from the cached generator — no name
# whitelist and no per-gate builder functions.
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=np.complex128)


def param_1q(theta: float, gen: np.ndarray) -> np.ndarray:
    """Build ``exp(-i θ P/2) = cos(θ/2) I − i sin(θ/2) P`` from a generator ``P``."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return c * I2 - 1j * s * gen


def extract_1q_generator(gate, parameter) -> np.ndarray:
    """
    Extract the Pauli generator ``P`` of a single-qubit rotation gate.

    Uses the identity ``U(π) = -i P``  ⇒  ``P = i·U(π)``, valid whenever the
    gate has the form ``exp(-i θ P/2)`` with ``P² = I`` — exactly the class of
    gates for which the π/2 parameter-shift rule is correct.

    Raises
    ------
    ValueError
        If the extracted generator is not Hermitian and involutory, i.e. the
        gate is not a pure Pauli-axis rotation and parameter-shift would be
        invalid.
    """
    qc = QuantumCircuit(1)
    qc.append(gate, [0])
    bound = qc.assign_parameters({parameter: np.pi})
    u_pi = np.asarray(bound.data[0].operation.to_matrix(), dtype=np.complex128)
    gen = 1j * u_pi

    if not np.allclose(gen, gen.conj().T, atol=1e-7):
        raise ValueError(
            f"Parameterized gate '{gate.name}' has a non-Hermitian generator; "
            "only Pauli-axis rotations (rx/ry/rz/…) can be parameterized."
        )
    if not np.allclose(gen @ gen, I2, atol=1e-7):
        raise ValueError(
            f"Parameterized gate '{gate.name}' generator is not involutory "
            "(P² ≠ I); the π/2 parameter-shift rule does not apply."
        )
    return gen
