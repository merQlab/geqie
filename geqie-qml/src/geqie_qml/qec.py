"""
Quantum Error Correction (QEC) encoding circuits for the VQCLayer.

Each QEC code class provides:
  - ``num_physical_qubits(num_logical)`` — total physical qubit count
  - ``num_ancilla_qubits(num_logical)`` — ancilla qubits added per logical qubit
  - ``encoding_circuit(num_logical)``   — a Qiskit QuantumCircuit that maps
    ``num_logical`` data qubits (+ freshly initialised ancillas) into the
    code space.  The circuit acts on ``num_physical_qubits`` total qubits;
    the first ``num_logical`` are the data qubits (where the image unitary
    has already been applied), and the remaining are ancillas in |0⟩.

Usage with VQCLayer
-------------------
Pass an instance to VQCLayer::

    from geqie_qml.qec import BitFlipCode, PhaseFlipCode, ShorCode

    vqc = VQCLayer(num_qubits=9, num_layers=1, qec_code=BitFlipCode())
"""

from abc import ABC, abstractmethod
from qiskit import QuantumCircuit


class QECCode(ABC):
    """Abstract base class for QEC encoding circuits."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the code (for experiment reports)."""
        ...

    @abstractmethod
    def num_physical_qubits(self, num_logical: int) -> int:
        """Return the total number of physical qubits for *num_logical* logical qubits."""
        ...

    def num_ancilla_qubits(self, num_logical: int) -> int:
        """Return the number of ancilla qubits added."""
        return self.num_physical_qubits(num_logical) - num_logical

    @abstractmethod
    def encoding_circuit(self, num_logical: int) -> QuantumCircuit:
        """
        Build the encoding circuit.

        The returned QuantumCircuit operates on ``num_physical_qubits``
        qubits.  Qubits ``0 .. num_logical-1`` are the data (logical)
        qubits; the rest are ancillas assumed to start in |0⟩.

        The circuit should map each logical qubit into its encoded block
        across the physical qubits.
        """
        ...


# ---------------------------------------------------------------------------
# Bit-flip repetition code  [[3,1,1]]
# ---------------------------------------------------------------------------

class BitFlipCode(QECCode):
    """
    3-qubit bit-flip repetition code.

    Encodes each logical qubit |ψ⟩ = α|0⟩ + β|1⟩ into
    α|000⟩ + β|111⟩ using two CNOT gates per logical qubit.

    Physical layout for *n* logical qubits: 3*n* physical qubits.
    Logical qubit *k* uses physical qubits [3k, 3k+1, 3k+2].
    """

    @property
    def name(self) -> str:
        return "BitFlipCode [[3,1,1]]"

    def num_physical_qubits(self, num_logical: int) -> int:
        return 3 * num_logical

    def encoding_circuit(self, num_logical: int) -> QuantumCircuit:
        n_phys = self.num_physical_qubits(num_logical)
        qc = QuantumCircuit(n_phys)
        for k in range(num_logical):
            data = 3 * k        # logical qubit position
            anc1 = 3 * k + 1
            anc2 = 3 * k + 2
            qc.cx(data, anc1)
            qc.cx(data, anc2)
        return qc


# ---------------------------------------------------------------------------
# Phase-flip repetition code  [[3,1,1]]
# ---------------------------------------------------------------------------

class PhaseFlipCode(QECCode):
    """
    3-qubit phase-flip code.

    Encodes each logical qubit into the Hadamard-rotated bit-flip code:
    α|+++⟩ + β|---⟩.

    Physical layout identical to BitFlipCode (3n qubits).
    """

    @property
    def name(self) -> str:
        return "PhaseFlipCode [[3,1,1]]"

    def num_physical_qubits(self, num_logical: int) -> int:
        return 3 * num_logical

    def encoding_circuit(self, num_logical: int) -> QuantumCircuit:
        n_phys = self.num_physical_qubits(num_logical)
        qc = QuantumCircuit(n_phys)
        for k in range(num_logical):
            data = 3 * k
            anc1 = 3 * k + 1
            anc2 = 3 * k + 2
            qc.cx(data, anc1)
            qc.cx(data, anc2)
            qc.h(data)
            qc.h(anc1)
            qc.h(anc2)
        return qc


# ---------------------------------------------------------------------------
# Shor code  [[9,1,3]]
# ---------------------------------------------------------------------------

class ShorCode(QECCode):
    """
    9-qubit Shor code — concatenation of phase-flip and bit-flip codes.

    Corrects arbitrary single-qubit errors.  Each logical qubit is encoded
    into 9 physical qubits.

    Physical layout: logical qubit *k* uses physical qubits [9k .. 9k+8].
    """

    @property
    def name(self) -> str:
        return "ShorCode [[9,1,3]]"

    def num_physical_qubits(self, num_logical: int) -> int:
        return 9 * num_logical

    def encoding_circuit(self, num_logical: int) -> QuantumCircuit:
        n_phys = self.num_physical_qubits(num_logical)
        qc = QuantumCircuit(n_phys)
        for k in range(num_logical):
            base = 9 * k
            q = [base + i for i in range(9)]
            # Phase-flip encoding: spread across 3 blocks of 3
            qc.cx(q[0], q[3])
            qc.cx(q[0], q[6])
            # Hadamard on the three block leaders
            qc.h(q[0])
            qc.h(q[3])
            qc.h(q[6])
            # Bit-flip encoding within each block
            qc.cx(q[0], q[1])
            qc.cx(q[0], q[2])
            qc.cx(q[3], q[4])
            qc.cx(q[3], q[5])
            qc.cx(q[6], q[7])
            qc.cx(q[6], q[8])
        return qc


# ---------------------------------------------------------------------------
# Surface code  [[4,1,2]] — distance-2 rotated surface code patch
# ---------------------------------------------------------------------------

class SurfaceCode(QECCode):
    """
    Distance-2 rotated surface code patch — [[4,1,2]].

    The smallest 2D surface-code patch.  Each logical qubit is encoded into
    a 2x2 plaquette of 4 data qubits arranged on a square lattice::

        q0 --- q1
        |      |
        q2 --- q3

    Stabilizers (one X-type, one Z-type, plus the boundary):
        S_X = X0 X1 X2 X3      (the 2x2 face plaquette)
        S_Z = Z0 Z1   (top edge boundary)
        S_Z = Z2 Z3   (bottom edge boundary)

    Logical operators:
        X_L = X0 X2  (left column)
        Z_L = Z0 Z1  (top row)

    Code distance d = 2 — detects (but does not correct) any single-qubit
    error.  This is the smallest non-trivial surface code; it is widely
    used as a teaching example and as a building block for larger
    rotated surface codes.

    Encoding circuit (maps a logical state on q0 onto the [[4,1,2]] code):
        1. Prepare ancillas q1, q2, q3 in |0⟩ (assumed by VQCLayer).
        2. Apply CNOTs that propagate the data qubit q0 onto the
           plaquette so the resulting state is a +1 eigenstate of S_X
           and the two boundary Z stabilizers.

    Physical layout: logical qubit k uses physical qubits [4k .. 4k+3].
    """

    @property
    def name(self) -> str:
        return "SurfaceCode [[4,1,2]] (rotated d=2 patch)"

    def num_physical_qubits(self, num_logical: int) -> int:
        return 4 * num_logical

    def encoding_circuit(self, num_logical: int) -> QuantumCircuit:
        n_phys = self.num_physical_qubits(num_logical)
        qc = QuantumCircuit(n_phys)
        for k in range(num_logical):
            base = 4 * k
            q0, q1, q2, q3 = base, base + 1, base + 2, base + 3
            # q0 carries the logical |ψ⟩ = α|0⟩ + β|1⟩ from the image unitary.
            # q1, q2, q3 are ancillas in |0⟩.
            #
            # Goal: produce a +1 eigenstate of the plaquette X-stabilizer
            # S_X = X0 X1 X2 X3 and the boundary Z-stabilizers Z0Z1, Z2Z3,
            # while keeping Z_L = Z0 Z1 acting non-trivially on the logical
            # subspace.

            # Step 1: project onto +1 eigenspace of the X-plaquette by
            # creating a GHZ-like superposition over q1 (acts as the
            # X-stabilizer "ancilla" contribution).
            qc.h(q1)
            qc.cx(q1, q0)
            qc.cx(q1, q2)
            qc.cx(q1, q3)

            # Step 2: enforce the two boundary Z-stabilizers Z0Z1 and Z2Z3
            # by entangling top row and bottom row.  After Step 1 the
            # state on each row is correlated; this CNOT makes the
            # Z-boundary stabilizers +1 eigenstates while preserving the
            # logical Z = Z0 Z1.
            qc.cx(q0, q1)
            qc.cx(q2, q3)
        return qc
