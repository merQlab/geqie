import numpy as np
from qiskit.quantum_info import Operator

KET_BRA_0 = np.array([[1, 0], [0, 0]])


def R_y_gate_kron_ket_0_bra_0(p):
    R_y = [[np.cos(p), -np.sin(p)],
           [np.sin(p), np.cos(p)]]
    return np.kron(R_y, KET_BRA_0)


def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    m = u * image.shape[0] + v
    p = image[u, v] * np.pi / (2 * 255.0) - np.pi / 4
    operator_dim = 2 ** (2 * R + 2)
    map_operator = np.zeros((4, operator_dim))
    block_start = 4 * m
    map_operator[:, block_start:block_start + 4] = R_y_gate_kron_ket_0_bra_0(p)
    return Operator(map_operator)
