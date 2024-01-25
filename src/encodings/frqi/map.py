import numpy as np

from qiskit.quantum_info import Statevector


def map(u: int, v: int, image: np.ndarray) -> Statevector:
    R = np.ceil(np.log2(np.max(image.shape))).astype(int)

    p = image[u, v]

    return Statevector([np.cos(np.pi/2*p), np.sin(np.pi/2*p)])