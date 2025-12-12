import numpy as np
import json

def retrieve(results: str) -> np.ndarray:
    """
    Decodes an image from quantum state measurement results.

    Parameters:
    results (dict): A dictionary where keys are binary strings representing quantum states,
                         and values are their respective occurrence counts.

    Returns:
    np.ndarray: A NumPy array representing the decoded image.
    """
    state_length = len(next(iter(results)))
    color_qubits = 10
    number_of_position_qubits = state_length - color_qubits
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2

    image_shape = (2**x_qubits, 2**y_qubits)
    reconstructed_image = np.zeros((image_shape[0], image_shape[1], 3))

    for state, n in results.items():
        if n > 0:
            x = state[0:x_qubits]
            y = state[y_qubits: number_of_position_qubits]
            c = state[number_of_position_qubits: number_of_position_qubits+color_qubits+1] # It's 10+1 because of python array indexing
            x_dec = int(x, base=2)
            y_dec = int(y, base=2)
            if c[8:10] == "00":
                reconstructed_image[x_dec, y_dec, 0] = int(c[0:8], base=2)
            elif c[8:10] == "01":
                reconstructed_image[x_dec, y_dec, 1] = int(c[0:8], base=2)
            elif c[8:10] == "10":
                reconstructed_image[x_dec, y_dec, 2] = int(c[0:8], base=2)
        
    return reconstructed_image.astype(np.uint8)
