import numpy as np
from qiskit.quantum_info import Statevector

def data(rho: int, theta: int, R: int, image: np.ndarray) -> Statevector:
    m = rho * image.shape[0] + theta
    data_vector = np.zeros(2**(2 * R))
    data_vector[m] = 1

    return Statevector(data_vector)