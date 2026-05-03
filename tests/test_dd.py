"""
Tests for dynamical-decoupling pulse insertion.

Builds a 4-qubit circuit with a deliberate idle window on qubit 1 (an
explicit ``Delay(1000)`` flanked by two ``X`` gates on qubit 0), runs
``apply_dd`` with the ``"XY4"`` sequence against a FakeBackend's target,
and verifies that qubit 1 now carries the four inserted ``X-Y-X-Y``
pulses while qubit 0's two original ``X`` gates are preserved.
"""

from __future__ import annotations

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import YGate
from qiskit.transpiler import InstructionProperties

from geqie_qml.noise_suppression.dynamical_decoupling import apply_dd


def _qubit_index(circuit: QuantumCircuit, qubit) -> int:
    """Return the position of ``qubit`` in ``circuit``'s quantum register."""
    return circuit.find_bit(qubit).index


def _ops_on_qubit(circuit: QuantumCircuit, q_index: int) -> list[str]:
    """List the names of single-qubit ops on ``q_index``, skipping delays/barriers."""
    return [
        instr.operation.name
        for instr in circuit.data
        if len(instr.qubits) == 1
        and _qubit_index(circuit, instr.qubits[0]) == q_index
        and instr.operation.name not in ("delay", "barrier")
    ]


def _fake_target_with_y():
    """
    Return ``backend.target`` from a FakeBackend, with ``YGate`` durations
    registered so the ``XY4`` DD sequence can be padded.

    Modern IBM fake backends (FakeBrisbane, FakeKyoto, ...) use the basis
    ``{ecr, id, rz, sx, x}`` and have no ``y`` instruction.  Adding it
    with the same duration as ``x`` is enough for the scheduler to size
    the DD slots correctly.
    """
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeBrisbane

    backend = FakeBrisbane()
    target = backend.target
    if "y" not in target.operation_names:
        y_props = {}
        for q in range(target.num_qubits):
            x_props = target["x"].get((q,))
            if x_props is None or x_props.duration is None:
                continue
            y_props[(q,)] = InstructionProperties(duration=x_props.duration)
        target.add_instruction(YGate(), y_props)
    return target


def test_apply_dd_inserts_xy4_on_idle_qubit():
    target = _fake_target_with_y()

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.delay(1000, 1, unit="dt")
    qc.x(0)

    out = apply_dd(qc, target, sequence_name="XY4")

    # Qubit 1 was idle for the full Delay(1000) window; with the default
    # PadDynamicalDecoupling settings (insert_multiple_cycles=False) that
    # window receives exactly one XY4 cycle: X, Y, X, Y.
    q1_ops = _ops_on_qubit(out, 1)
    assert q1_ops == ["x", "y", "x", "y"], (
        f"Expected X-Y-X-Y on qubit 1 from XY4, got {q1_ops}"
    )

    # Qubit 0's two original X gates must still be present in the output.
    # (PadDynamicalDecoupling also fills q0's own idle window between those
    # two X gates with a DD cycle, so q0 is not literally byte-identical to
    # the input — but the user-placed gates survive untouched.)
    q0_ops = _ops_on_qubit(out, 0)
    assert q0_ops.count("x") >= 2, (
        f"q0's two original X gates should be preserved, got {q0_ops}"
    )


def test_apply_dd_rejects_none_target():
    qc = QuantumCircuit(2)
    qc.x(0)
    with pytest.raises(ValueError, match="duration"):
        apply_dd(qc, None, sequence_name="XY4")


def test_apply_dd_rejects_unknown_sequence():
    target = _fake_target_with_y()
    qc = QuantumCircuit(2)
    qc.x(0)
    with pytest.raises(ValueError, match="Unknown DD sequence"):
        apply_dd(qc, target, sequence_name="not-a-real-sequence")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
