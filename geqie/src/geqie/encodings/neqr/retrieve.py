import numpy as np
from typing import Any

def retrieve(results: dict[str, int], bitrate:int=8, **_: Any) -> np.ndarray:
    """
    Decodes an image from quantum state measurement results.

    Parameters:
    results (dict): A dictionary where keys are binary strings representing quantum states,
                         and values are their respective occurrence counts.
    bitrate (int): The number of bits used to represent each pixel value.

    Returns:
    np.ndarray: A NumPy array representing the decoded image.
    """
    state_length = len(next(iter(results)))
    number_of_position_qubits = state_length - bitrate
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2

    image_shape = (2**x_qubits, 2**y_qubits)
    reconstructed_image = np.zeros((image_shape[0], image_shape[1]))

    for state, n in results.items():
        if n > 0:
            x = state[0:x_qubits]
            y = state[x_qubits:number_of_position_qubits]

            c = state[number_of_position_qubits:number_of_position_qubits+bitrate]

            x_dec = int(x, base=2)
            y_dec = int(y, base=2)
            c_dec = int(c, base=2)
            reconstructed_image[x_dec, y_dec] = c_dec
        
    return reconstructed_image.astype(np.uint8)
