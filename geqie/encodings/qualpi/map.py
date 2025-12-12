import numpy as np

from qiskit.quantum_info import Operator

I_GATE = np.eye(2)
X_GATE = np.array([
    [0, 1], 
    [1, 0]
])


def map(rho: int, theta: int, R: int, image: np.ndarray) -> Operator:
    p = image[rho, theta]
    # Convert value to binary string, without '0b' and padded with 0s, e.g.: '0001 1101':
    pixel_value_as_binary_string = bin(p)[2:].zfill(8)
    # Convert to logic array:
    pixel_value_as_binary_array = [int(bit) for bit in pixel_value_as_binary_string][::-1]

    if pixel_value_as_binary_array[0] == 1:
        map_operator = X_GATE
    else:
        map_operator = I_GATE

    for bit in pixel_value_as_binary_array[1:8]:
        if bit == 1:
            map_operator = np.kron(X_GATE, map_operator)
        else:
            map_operator = np.kron(I_GATE, map_operator)

    return Operator(map_operator)
