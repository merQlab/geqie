import numpy as np
from qiskit.quantum_info import Statevector


def init(n_qubits: int) -> Statevector:
    """
    NAQSS initial state.

    We simply use |0...0>, which matches the initial state |0⊗(n+1)⟩
    used to build the NAQSS state in Li et al. (Eq. (18)).
    """
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return Statevector(state)
