import numpy as np

from qiskit.quantum_info import Statevector


def init(n_qubits: int) -> Statevector:
    state = np.tile([1, 0], 2**(n_qubits-4)) # np.tile([1, 0], 2**(n_qubits-4))
    return Statevector(state)
