import numpy as np

from qiskit.quantum_info import Statevector


def data(*coords: int, R: int, image: np.ndarray) -> Statevector:
    n_dims = len(coords)
    m = np.ravel_multi_index(coords, image.shape)
    data_vector = np.zeros(2**(n_dims * R))
    data_vector[m] = 1

    return Statevector(data_vector)
