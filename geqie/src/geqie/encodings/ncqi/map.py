import numpy as np

from qiskit.quantum_info import Operator


I_GATE = np.eye(2)
X_GATE = np.array([
    [0, 1], 
    [1, 0]
])


def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    map_operator = [None, None, None]

    for channel in range(3):
        # Red channel:
        p = image[u, v, channel]
        # Convert value to string to the binary form, cut '0b', and padd with 0 example: '0001 1101':
        pixel_value_as_binary_string = bin(p)[2:].zfill(8)
        # Convert to logic array:
        pixel_value_as_binary_array = [int(bit) for bit in pixel_value_as_binary_string][::-1]

        if channel == 0:
            map_operator[channel] = np.kron(I_GATE, I_GATE)
        elif channel == 1:
            map_operator[channel] = np.kron(I_GATE, X_GATE)
        elif channel == 2:
            map_operator[channel] = np.kron(X_GATE, I_GATE)

        for bit in pixel_value_as_binary_array[0:8]:
            if bit == 1:
                map_operator[channel] = np.kron(X_GATE, map_operator[channel])
            else:
                map_operator[channel] = np.kron(I_GATE, map_operator[channel])

    map_operator = map_operator[0] + map_operator[1] + map_operator[2]

    return Operator(map_operator)
