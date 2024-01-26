import numpy as np

from qiskit.quantum_info import Statevector


def map(u: int, v: int, R: int, image: np.ndarray) -> Statevector:
    p = image[u, v]

    return Statevector([np.cos(np.pi/2*p), np.sin(np.pi/2*p)])