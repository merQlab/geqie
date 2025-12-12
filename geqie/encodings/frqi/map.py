import numpy as np

from qiskit.quantum_info import Operator


def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    p = image[u, v] / 255.0 * (np.pi / 2)
    map_operator = [
        [np.cos(p), -np.sin(p)],
        [np.sin(p),  np.cos(p)],
    ]

    return Operator(map_operator)
