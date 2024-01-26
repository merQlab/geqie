import numpy as np

from qiskit.quantum_info import Statevector


def data(u: int, v: int, R: int, image: np.ndarray) -> Statevector:
    m = u * image.shape[0] + v
    data_vector = np.zeros(2**(2*R))
    data_vector[m] = 1

    return Statevector(data_vector)