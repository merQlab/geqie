import numpy as np

from qiskit.quantum_info import Operator

identity_gate = np.array([[1, 0], [0, 1]])
x_gate = np.array([[0, 1], [1, 0]])


def map(rho: int, theta: int, R: int, image: np.ndarray) -> Operator:
    p = image[rho, theta]
    # Convert value to string to the binary form, cut '0b', and padd with 0 example: '0001 1101':
    pixel_value_as_binary_string = bin(p)[2:].zfill(8)
    # Convert to logic array:
    pixel_value_as_binary_array = [int(bit) for bit in pixel_value_as_binary_string][::-1]

    if pixel_value_as_binary_array[0] == 1:
        map_operator = x_gate
    else:
        map_operator = identity_gate

    for bit in pixel_value_as_binary_array[1:8]:
        if bit == 1:
            map_operator = np.kron(x_gate, map_operator)
        else:
            map_operator = np.kron(identity_gate, map_operator)

    return Operator(map_operator)
