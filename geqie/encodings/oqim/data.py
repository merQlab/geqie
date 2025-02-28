import numpy as np
from qiskit.quantum_info import Statevector, Operator

KET_BRA_1 = np.array([[0, 0], [0, 1]])


def R_y_gate_kron_ket_1_bra_1(p):
    R_y = [[np.cos(p), -np.sin(p)],
           [np.sin(p), np.cos(p)]]
    return np.kron(R_y, KET_BRA_1)


def data(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    m = u * image.shape[0] + v
    total_pixels = image.shape[0] * image.shape[1] - 1
    pixel_position_encoded_as_angle = ((m / total_pixels) * np.pi / 2) - np.pi / 4
    operator_dim = 2 ** (2 * R + 2)
    data_operator = np.zeros((4, operator_dim))
    block_start = 4 * m
    data_operator[:, block_start: block_start + 4] = R_y_gate_kron_ket_1_bra_1(pixel_position_encoded_as_angle)
    return Operator(data_operator)
