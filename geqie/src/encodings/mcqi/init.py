import numpy as np
from qiskit.quantum_info import Statevector

def init(n_qubits: int) -> Statevector:
    base_state = np.zeros(2**3, dtype=int)
    base_state[0] = 1
    state = np.tile(base_state, 2**(n_qubits - 3))
    return Statevector(state)
