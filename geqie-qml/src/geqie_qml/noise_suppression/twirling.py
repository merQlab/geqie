"""
Pauli twirling of two-qubit entangling gates.

Pauli twirling is a noise-suppression technique that converts coherent
two-qubit gate noise into a stochastic Pauli channel.  For an entangling
gate ``G``, twirling replaces every occurrence of ``G`` by

    P_pre · G · P_post

where ``P_pre = P_c ⊗ P_t`` is a uniformly random Pauli pair drawn from
{I, X, Y, Z}^2 and ``P_post = (G · P_pre · G†)`` is the unique Pauli pair
(up to global phase) that undoes ``P_pre`` in the noiseless limit.  The
sandwich therefore equals ``G`` (up to a global phase) for every draw, so
the noiseless action is preserved; under coherent noise on ``G`` the
average over many draws projects the error onto its Pauli-diagonal part.

This module exposes:

* ``PAULI_TWIRL_CNOT`` / ``PAULI_TWIRL_CZ`` / ``PAULI_TWIRL_ECR`` — the
  16-entry conjugation tables for the supported gates.  Each maps
  ``(P_c, P_t) -> (P_c_out, P_t_out)`` such that
  ``(P_c_out ⊗ P_t_out) · G · (P_c ⊗ P_t) = G`` up to global phase.
* ``twirl_two_qubit_gate(qc, gate_name, qubits, rng)`` — append a
  randomly twirled instance of ``gate_name`` to ``qc``.
* ``twirl_circuit(qc, rng, gates_to_twirl=...)`` — return a NEW
  ``QuantumCircuit`` with every two-qubit gate in ``gates_to_twirl``
  replaced by its twirled equivalent.

The CX table is the canonical one derived from the standard CNOT
conjugation rules::

    CX (I⊗I) CX = I⊗I       CX (I⊗X) CX = I⊗X
    CX (X⊗I) CX = X⊗X       CX (Z⊗I) CX = Z⊗I
    CX (I⊗Z) CX = Z⊗Z       CX (Y⊗I) CX = Y⊗X
    CX (I⊗Y) CX = Z⊗Y       CX (X⊗X) CX = X⊗I
    CX (Y⊗Y) CX = X⊗Z       CX (Z⊗Z) CX = I⊗Z      ...

with the remaining entries fixed by component-wise multiplication of the
above (the conjugation is a homomorphism on the 2-qubit Pauli group up to
phase).  CZ and ECR are derived analogously; see
``_compute_twirl_table`` for the construction.
"""

from __future__ import annotations

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import CXGate, CZGate, ECRGate
from qiskit.quantum_info import Operator, Pauli


_PAULI_LETTERS: tuple[str, str, str, str] = ("I", "X", "Y", "Z")


# ---------------------------------------------------------------------------
# Hand-derived twirl tables for documentation and direct inspection
# ---------------------------------------------------------------------------
# CNOT (control = qubit 0, target = qubit 1).  Derived from the action of
# CX on the 2-qubit Pauli group: c_P(P_c) = CX(P_c⊗I)CX, t_P(P_t) = CX(I⊗P_t)CX,
# and CX(P_c⊗P_t)CX = c_P(P_c) · t_P(P_t).  All 16 entries are filled in.

PAULI_TWIRL_CNOT: dict[tuple[str, str], tuple[str, str]] = {
    ("I", "I"): ("I", "I"),
    ("I", "X"): ("I", "X"),
    ("I", "Y"): ("Z", "Y"),
    ("I", "Z"): ("Z", "Z"),
    ("X", "I"): ("X", "X"),
    ("X", "X"): ("X", "I"),
    ("X", "Y"): ("Y", "Z"),
    ("X", "Z"): ("Y", "Y"),
    ("Y", "I"): ("Y", "X"),
    ("Y", "X"): ("Y", "I"),
    ("Y", "Y"): ("X", "Z"),
    ("Y", "Z"): ("X", "Y"),
    ("Z", "I"): ("Z", "I"),
    ("Z", "X"): ("Z", "X"),
    ("Z", "Y"): ("I", "Y"),
    ("Z", "Z"): ("I", "Z"),
}

# CZ (symmetric).  Derived from cz_c(P_c) = CZ(P_c⊗I)CZ and
# cz_t(P_t) = CZ(I⊗P_t)CZ:
#     cz_c: I->(I,I)  X->(X,Z)  Y->(Y,Z)  Z->(Z,I)
#     cz_t: I->(I,I)  X->(Z,X)  Y->(Z,Y)  Z->(I,Z)
# combined by component-wise Pauli multiplication.

PAULI_TWIRL_CZ: dict[tuple[str, str], tuple[str, str]] = {
    ("I", "I"): ("I", "I"),
    ("I", "X"): ("Z", "X"),
    ("I", "Y"): ("Z", "Y"),
    ("I", "Z"): ("I", "Z"),
    ("X", "I"): ("X", "Z"),
    ("X", "X"): ("Y", "Y"),
    ("X", "Y"): ("Y", "X"),
    ("X", "Z"): ("X", "I"),
    ("Y", "I"): ("Y", "Z"),
    ("Y", "X"): ("X", "Y"),
    ("Y", "Y"): ("X", "X"),
    ("Y", "Z"): ("Y", "I"),
    ("Z", "I"): ("Z", "I"),
    ("Z", "X"): ("I", "X"),
    ("Z", "Y"): ("I", "Y"),
    ("Z", "Z"): ("Z", "Z"),
}


# ---------------------------------------------------------------------------
# Programmatic derivation (used to populate PAULI_TWIRL_ECR and to validate
# the hand-typed CX/CZ tables at import time).
# ---------------------------------------------------------------------------

def _two_qubit_pauli_matrix(p_c: str, p_t: str) -> np.ndarray:
    """
    Return the 4x4 matrix of (P_c ⊗ P_t) where P_c is on the control qubit
    (index 0) and P_t is on the target qubit (index 1), in qiskit's
    little-endian convention (qubit 0 is the LSB tensor factor).
    """
    return np.kron(Pauli(p_t).to_matrix(), Pauli(p_c).to_matrix())


def _identify_pauli(matrix: np.ndarray) -> tuple[str, str]:
    """
    Return (P_c, P_t) such that ``matrix == phase * (P_c ⊗ P_t)`` for some
    unit-modulus complex phase.
    """
    for p_c in _PAULI_LETTERS:
        for p_t in _PAULI_LETTERS:
            ref = _two_qubit_pauli_matrix(p_c, p_t)
            idx = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
            if abs(ref[idx]) < 1e-12:
                continue
            phase = matrix[idx] / ref[idx]
            if np.allclose(matrix, phase * ref, atol=1e-9):
                return (p_c, p_t)
    raise ValueError("Conjugated operator is not a 2-qubit Pauli up to phase.")


def _compute_twirl_table(gate_unitary: np.ndarray) -> dict[tuple[str, str], tuple[str, str]]:
    """
    Compute the Pauli-twirl conjugation table for a given Clifford 2-qubit
    gate ``G``.  For every (P_c, P_t) returns the (P_c_out, P_t_out) such
    that ``G · (P_c ⊗ P_t) · G† = phase * (P_c_out ⊗ P_t_out)``.
    """
    table: dict[tuple[str, str], tuple[str, str]] = {}
    g_dag = gate_unitary.conj().T
    for p_c in _PAULI_LETTERS:
        for p_t in _PAULI_LETTERS:
            pre = _two_qubit_pauli_matrix(p_c, p_t)
            post = gate_unitary @ pre @ g_dag
            table[(p_c, p_t)] = _identify_pauli(post)
    return table


# Sanity-check the hand-typed CX and CZ tables.  These asserts run once at
# import time and protect against typos in the literal dicts above.
_CX_MATRIX = np.asarray(Operator(CXGate()).data, dtype=complex)
_CZ_MATRIX = np.asarray(Operator(CZGate()).data, dtype=complex)
_ECR_MATRIX = np.asarray(Operator(ECRGate()).data, dtype=complex)

assert _compute_twirl_table(_CX_MATRIX) == PAULI_TWIRL_CNOT, (
    "PAULI_TWIRL_CNOT does not match CX conjugation; the hand-typed table is wrong."
)
assert _compute_twirl_table(_CZ_MATRIX) == PAULI_TWIRL_CZ, (
    "PAULI_TWIRL_CZ does not match CZ conjugation; the hand-typed table is wrong."
)

# ECR is locally Clifford-equivalent to CX but its Pauli action is less
# easy to read off by hand, so populate it programmatically.
PAULI_TWIRL_ECR: dict[tuple[str, str], tuple[str, str]] = _compute_twirl_table(_ECR_MATRIX)


# ---------------------------------------------------------------------------
# Twirling primitives
# ---------------------------------------------------------------------------

_GATE_TABLES: dict[str, dict[tuple[str, str], tuple[str, str]]] = {
    "cx": PAULI_TWIRL_CNOT,
    "cz": PAULI_TWIRL_CZ,
    "ecr": PAULI_TWIRL_ECR,
}


def _emit_pauli(qc: QuantumCircuit, pauli: str, qubit) -> None:
    """Append the named Pauli (one of I/X/Y/Z) on a single qubit."""
    if pauli == "I":
        qc.id(qubit)
    elif pauli == "X":
        qc.x(qubit)
    elif pauli == "Y":
        qc.y(qubit)
    elif pauli == "Z":
        qc.z(qubit)
    else:
        raise ValueError(f"Unknown Pauli letter {pauli!r}; expected one of I, X, Y, Z.")


def _emit_native_two_qubit_gate(qc: QuantumCircuit, gate_name: str, qubits) -> None:
    """Append the named entangling gate using qiskit's standard method."""
    if gate_name == "cx":
        qc.cx(qubits[0], qubits[1])
    elif gate_name == "cz":
        qc.cz(qubits[0], qubits[1])
    elif gate_name == "ecr":
        qc.ecr(qubits[0], qubits[1])
    else:
        raise ValueError(
            f"Unsupported entangling gate {gate_name!r}; "
            f"supported: {sorted(_GATE_TABLES)}."
        )


def twirl_two_qubit_gate(qc: QuantumCircuit, gate_name: str, qubits, rng) -> None:
    """
    Append a randomly twirled version of the named two-qubit gate to ``qc``.

    Picks a uniform random ``(P_c, P_t)`` from {I, X, Y, Z}^2, looks up the
    corresponding ``(P_c_out, P_t_out)`` in the gate's conjugation table,
    and emits

        P_c on qubits[0], P_t on qubits[1],     # pre-Paulis
        gate(qubits[0], qubits[1]),
        P_c_out on qubits[0], P_t_out on qubits[1].   # post-Paulis

    The Pauli operations are emitted as standard Qiskit gates (``id``,
    ``x``, ``y``, ``z``).

    Parameters
    ----------
    qc : QuantumCircuit
        Circuit to append onto.
    gate_name : str
        One of ``"cx"``, ``"cz"``, ``"ecr"`` (case-insensitive).
    qubits : sequence of length 2
        ``(control, target)`` qubit specifiers — anything ``qc.cx`` etc. accepts.
    rng : numpy.random.Generator
        Source of randomness.  Use a seeded ``numpy.random.default_rng(seed)``
        for reproducibility.
    """
    name = gate_name.lower()
    table = _GATE_TABLES.get(name)
    if table is None:
        raise ValueError(
            f"Unsupported entangling gate {gate_name!r}; "
            f"supported: {sorted(_GATE_TABLES)}."
        )
    if len(qubits) != 2:
        raise ValueError(f"Two-qubit gate {gate_name!r} requires 2 qubits, got {len(qubits)}.")

    p_c = _PAULI_LETTERS[int(rng.integers(4))]
    p_t = _PAULI_LETTERS[int(rng.integers(4))]
    p_c_out, p_t_out = table[(p_c, p_t)]

    _emit_pauli(qc, p_c, qubits[0])
    _emit_pauli(qc, p_t, qubits[1])
    _emit_native_two_qubit_gate(qc, name, qubits)
    _emit_pauli(qc, p_c_out, qubits[0])
    _emit_pauli(qc, p_t_out, qubits[1])


def twirl_circuit(
    qc: QuantumCircuit,
    rng,
    gates_to_twirl: tuple[str, ...] = ("cx", "cz", "ecr"),
) -> QuantumCircuit:
    """
    Return a NEW QuantumCircuit with every two-qubit gate in
    ``gates_to_twirl`` replaced by its randomly twirled equivalent.

    Walks ``qc.data`` in order; gates whose name (lowercased) is in
    ``gates_to_twirl`` are replaced by ``twirl_two_qubit_gate(...)``,
    while every other instruction (single-qubit gates, parametric gates,
    barriers, measurements, resets, classical control, etc.) is copied
    unchanged.  Quantum and classical registers, parameters, and global
    phase are preserved.

    Parameters
    ----------
    qc : QuantumCircuit
    rng : numpy.random.Generator
        Determines the random Pauli draws — the same rng/seed yields the
        same twirled circuit.
    gates_to_twirl : tuple of str
        Lowercased gate names to twirl.  Default covers ``cx``, ``cz``, ``ecr``.
    """
    targets = tuple(g.lower() for g in gates_to_twirl)
    new_qc = qc.copy_empty_like()
    for instruction in qc.data:
        op = instruction.operation
        qubits = instruction.qubits
        clbits = instruction.clbits
        if op.name.lower() in targets:
            twirl_two_qubit_gate(new_qc, op.name, qubits, rng)
        else:
            new_qc.append(op, qubits, clbits)
    return new_qc
