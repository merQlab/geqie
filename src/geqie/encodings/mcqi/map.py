import numpy as np
from qiskit.quantum_info import Operator


I_GATE = np.eye(2)
X_GATE = np.array([
    [0, 1], 
    [1, 0]
])

CHANNEL_POSITIONING = {
    0: np.kron(I_GATE, I_GATE),  # red
    1: np.kron(I_GATE, X_GATE),  # green
    2: np.kron(X_GATE, I_GATE),  # blue
}


def ry_gate(theta: float) -> np.ndarray:
    return [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ]


def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    operators = []
    for channel in range(3):
        p = image[u, v, channel] / 255.0 * (np.pi / 2)
        operators.append(np.kron(CHANNEL_POSITIONING[channel], ry_gate(p)))

    map_operator = np.array(operators).sum(axis=0)

    return Operator(map_operator)
