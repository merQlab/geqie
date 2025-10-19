import numpy as np
from qiskit.quantum_info import Operator

BIT_PAIR_ANGLES = {
    (0, 0): 0,
    (0, 1): 2 / 10 * np.pi,
    (1, 0): 3 / 10 * np.pi,
    (1, 1): 5 / 10 * np.pi,
}


def ry_gate(theta: float) -> np.ndarray:
    return [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]


def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    p = image[u, v]
    pixel_value_as_binary = [int(bit) for bit in bin(p)[2:].zfill(8)][::-1]

    # Initialize the map operator with the first bit pair
    bit_pair = tuple(pixel_value_as_binary[:2])
    map_operator = ry_gate(BIT_PAIR_ANGLES[bit_pair])

    # Iterate over remaining bit pairs and build the Kronecker product
    for k in range(2, 8, 2):
        bit_pair = tuple(pixel_value_as_binary[k : k + 2])
        map_operator = np.kron(ry_gate(BIT_PAIR_ANGLES[bit_pair]), map_operator)

    return Operator(map_operator)
