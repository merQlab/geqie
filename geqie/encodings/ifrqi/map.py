import numpy as np
from qiskit.quantum_info import Operator

angle_00 = 0
angle_01 = 2 * np.pi / 10
angle_10 = 3 * np.pi / 10
angle_11 = 5 * np.pi / 10


def R_y_gate(p):
    R_y = [[np.cos(p), -np.sin(p)],
           [np.sin(p), np.cos(p)]]
    return R_y


def get_angle(bit_pair):
    """Returns the angle corresponding to the given bit pair."""
    angles = {
        (0, 0): angle_00,
        (0, 1): angle_01,
        (1, 0): angle_10,
        (1, 1): angle_11
    }
    return angles[bit_pair]


def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    p = image[u, v]
    pixel_value_as_binary_array = [int(bit) for bit in bin(p)[2:].zfill(8)][::-1]

    # Initialize the map operator with the first bit pair
    map_operator = R_y_gate(get_angle(tuple(pixel_value_as_binary_array[:2])))

    # Iterate over remaining bit pairs and build the Kronecker product
    for k in range(2, 8, 2):
        bit_pair = tuple(pixel_value_as_binary_array[k:k + 2])
        map_operator = np.kron(R_y_gate(get_angle(bit_pair)), map_operator)

    return Operator(map_operator)
