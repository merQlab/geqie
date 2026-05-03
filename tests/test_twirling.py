"""
Tests for Pauli twirling of two-qubit entangling gates.

The defining property of twirling is that the noiseless action of the
gate is unchanged: for every random Pauli draw,
    P_post · G · P_pre  ==  G    (up to global phase).
This test exercises that property over 1000 distinct random draws on a
two-qubit CX circuit, simulating each twirled circuit via
``qiskit.quantum_info.Statevector`` and comparing the resulting unitary
to the original CX up to a global phase.
"""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from geqie_qml.noise_suppression.twirling import (
    PAULI_TWIRL_CNOT,
    PAULI_TWIRL_CZ,
    PAULI_TWIRL_ECR,
    twirl_circuit,
)


def _unitary_via_statevector(qc: QuantumCircuit) -> np.ndarray:
    """
    Reconstruct the full 2^n x 2^n unitary of ``qc`` by evolving every
    computational basis state through it via ``Statevector``.  The
    columns of the result are the evolved basis states.
    """
    n = qc.num_qubits
    dim = 2 ** n
    cols = []
    for i in range(dim):
        sv = Statevector.from_int(i, dim).evolve(qc)
        cols.append(sv.data)
    return np.column_stack(cols)


def _equal_up_to_global_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-10) -> bool:
    """True iff ``a == phase * b`` for some unit-modulus complex ``phase``."""
    flat_b = b.ravel()
    idx = int(np.argmax(np.abs(flat_b)))
    if abs(flat_b[idx]) < 1e-12:
        return np.allclose(a, 0.0, atol=atol)
    phase = a.ravel()[idx] / flat_b[idx]
    if abs(abs(phase) - 1.0) > 1e-9:
        return False
    return np.allclose(a, phase * b, atol=atol)


def test_twirl_circuit_preserves_cx_unitary():
    """Twirling a single-CX circuit must reproduce CX up to global phase
    for every one of 1000 distinct random seeds."""
    base = QuantumCircuit(2)
    base.cx(0, 1)
    base_unitary = _unitary_via_statevector(base)

    for seed in range(1000):
        rng = np.random.default_rng(seed)
        twirled = twirl_circuit(base, rng)
        twirled_unitary = _unitary_via_statevector(twirled)
        assert _equal_up_to_global_phase(twirled_unitary, base_unitary), (
            f"seed {seed}: twirled unitary differs from CX beyond a global phase"
        )


def test_twirl_circuit_preserves_cz_unitary():
    """Same property must hold for CZ."""
    base = QuantumCircuit(2)
    base.cz(0, 1)
    base_unitary = _unitary_via_statevector(base)

    for seed in range(200):
        rng = np.random.default_rng(seed)
        twirled = twirl_circuit(base, rng)
        twirled_unitary = _unitary_via_statevector(twirled)
        assert _equal_up_to_global_phase(twirled_unitary, base_unitary), (
            f"seed {seed}: CZ twirl differs beyond global phase"
        )


def test_twirl_circuit_preserves_ecr_unitary():
    """Same property must hold for ECR."""
    base = QuantumCircuit(2)
    base.ecr(0, 1)
    base_unitary = _unitary_via_statevector(base)

    for seed in range(200):
        rng = np.random.default_rng(seed)
        twirled = twirl_circuit(base, rng)
        twirled_unitary = _unitary_via_statevector(twirled)
        assert _equal_up_to_global_phase(twirled_unitary, base_unitary), (
            f"seed {seed}: ECR twirl differs beyond global phase"
        )


def test_twirl_tables_have_16_entries_each():
    """Sanity: every Pauli pair must be represented in the conjugation tables."""
    expected_keys = {(p, q) for p in "IXYZ" for q in "IXYZ"}
    assert set(PAULI_TWIRL_CNOT) == expected_keys
    assert set(PAULI_TWIRL_CZ) == expected_keys
    assert set(PAULI_TWIRL_ECR) == expected_keys
    for table in (PAULI_TWIRL_CNOT, PAULI_TWIRL_CZ, PAULI_TWIRL_ECR):
        for value in table.values():
            assert len(value) == 2 and value[0] in "IXYZ" and value[1] in "IXYZ"


def test_twirl_circuit_preserves_single_qubit_gates_and_measurements():
    """Non-target instructions (Rx, H, measure, barrier, reset) must pass through unchanged."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.rx(0.7, 1)
    qc.barrier()
    qc.cx(0, 1)
    qc.barrier()
    qc.measure(0, 0)
    qc.measure(1, 1)

    rng = np.random.default_rng(0)
    twirled = twirl_circuit(qc, rng)

    # Every original non-CX op must appear in the twirled circuit, in order.
    untouched = [instr for instr in qc.data if instr.operation.name != "cx"]
    matched = [instr for instr in twirled.data
               if instr.operation.name in {"h", "rx", "barrier", "measure"}]
    assert len(matched) == len(untouched)
    for orig, new in zip(untouched, matched):
        assert orig.operation.name == new.operation.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
