import numpy as np

from qiskit.quantum_info import Statevector


def data(u: int, v: int, image: np.ndarray) -> Statevector:
    R = np.ceil(np.log2(np.max(image.shape))).astype(int)

    m = u * image.shape[0] + v

    return Statevector([int(b) for b in reversed(f"{m:0{2*R}b}")])