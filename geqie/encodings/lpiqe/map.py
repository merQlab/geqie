import numpy as np
from qiskit.quantum_info import Operator

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    p = np.pi * image[u, v] / 255.0
    map_operator = [
        [np.exp(1j * p), 0],
        [0, np.exp(1j * p)],
    ]

    return Operator(map_operator)