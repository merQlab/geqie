import numpy as np
from qiskit.quantum_info import Statevector

had_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

def init(n_qubits: int) -> Statevector:
    state = np.tile([1], 2 ** n_qubits)

    return Statevector(state)
