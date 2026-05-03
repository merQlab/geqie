"""
Dynamical decoupling (DD) for VQC circuits.

Dynamical decoupling inserts sequences of single-qubit pulses into idle
windows of a scheduled circuit to refocus low-frequency / non-Markovian
noise — primarily dephasing, slow drift in single-qubit Hamiltonians, and
coherent control errors that average to zero over a properly chosen
pulse train.

DD is applied AFTER any noise injection or Pauli twirling, immediately
before the circuit is handed to a simulator/backend that respects gate
durations.  It is a no-op (and in fact actively meaningless) for
duration-less simulators such as ``StatevectorSampler``: those run the
circuit gate-by-gate without a physical timeline, so there is no idle
window for DD to fill.  Likewise, DD does not belong in ``precompute.py``,
which builds noiseless reference unitaries — there is no noise to
suppress on a mathematical object.

Pulse-sequence references
-------------------------
* XX  : a CPMG-like 2-pulse train.  Refocuses single-axis dephasing only.
* XY4 : the standard 4-pulse universal sequence.  Refocuses arbitrary
        single-qubit static noise (any combination of bit-flip and
        phase-flip terms) to leading order.
* KDD : the 5-pulse Knill DD-X composite of Souza, Alvarez, Suter,
        *Phys. Rev. Lett.* 106, 240501 (2011).  Phases
        :math:`\\phi_k = (\\pi/6,\\, 0,\\, \\pi/2,\\, 0,\\, \\pi/6)` give
        :math:`\\pi`-rotations about ``cos(\\phi)·X + sin(\\phi)·Y``,
        encoded here as ``RGate(\\pi, \\phi)``.  Two of the five entries
        reduce to plain ``X`` (``\\phi = 0``) and one to plain ``Y``
        (``\\phi = \\pi/2``); the two ``\\phi = \\pi/6`` rotations are
        the additional ingredient that gives KDD its higher-order
        robustness against pulse-amplitude and timing errors.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate, YGate, RGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_ibm_runtime.transpiler.passes.scheduling import PadDynamicalDecoupling


# Souza-Alvarez-Suter Knill DD-X composite: π-rotations with phases
# (π/6, 0, π/2, 0, π/6) about axes in the X-Y plane.
_KDD_PHASES = (math.pi / 6, 0.0, math.pi / 2, 0.0, math.pi / 6)
_KDD_SEQUENCE = [RGate(math.pi, phi) for phi in _KDD_PHASES]


DD_SEQUENCES: dict[str, list] = {
    "XX":  [XGate(), XGate()],
    "XY4": [XGate(), YGate(), XGate(), YGate()],
    "KDD": _KDD_SEQUENCE,
}


_DD_REQUIRES_TARGET_MSG = (
    "Dynamical decoupling requires per-gate duration information, which a "
    "duration-less simulator (e.g. StatevectorSampler) does not provide. "
    "Pass `backend.target` from a fake or real IBM backend "
    "(e.g. FakeKyoto, FakeBrisbane, ibm_brisbane) so the scheduler can "
    "compute idle windows and the lengths of the inserted DD pulses."
)


def make_dd_pass_manager(backend_target, sequence_name: str = "XY4") -> PassManager:
    """
    Build a PassManager that ALAP-schedules its input circuit and pads
    every idle window with the named DD sequence.

    Parameters
    ----------
    backend_target : qiskit.transpiler.Target
        Backend target with gate-duration information (typically
        ``backend.target`` of a fake or real IBM backend).  ``None`` is
        rejected with a clear ``ValueError`` because DD has no defined
        meaning without durations.
    sequence_name : str
        One of the keys of :data:`DD_SEQUENCES`.

    Returns
    -------
    PassManager
        Two-stage pipeline: ``ALAPScheduleAnalysis`` then
        ``PadDynamicalDecoupling`` on the supplied target.
    """
    if backend_target is None:
        raise ValueError(_DD_REQUIRES_TARGET_MSG)
    if sequence_name not in DD_SEQUENCES:
        raise ValueError(
            f"Unknown DD sequence {sequence_name!r}; "
            f"supported: {sorted(DD_SEQUENCES)}."
        )
    sequence = DD_SEQUENCES[sequence_name]

    return PassManager([
        ALAPScheduleAnalysis(target=backend_target),
        PadDynamicalDecoupling(target=backend_target, dd_sequence=sequence),
    ])


def apply_dd(
    circuit: QuantumCircuit,
    backend_target,
    sequence_name: str = "XY4",
) -> QuantumCircuit:
    """Run :func:`make_dd_pass_manager` on ``circuit`` and return the result."""
    pm = make_dd_pass_manager(backend_target, sequence_name=sequence_name)
    return pm.run(circuit)
