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
    color_qubits = 8
    number_of_position_qubits = state_length - color_qubits
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2
    
    image_shape = (2**x_qubits, 2**y_qubits)

    reconstructed_image = np.zeros((image_shape[0], image_shape[1]))

    for state, n in results.items():
        if n > 0:
            x = state[0:x_qubits]
            y = state[x_qubits:number_of_position_qubits]

            c = state[number_of_position_qubits:number_of_position_qubits+color_qubits+1]

            x_dec = int(x, base=2)
            y_dec = int(y, base=2)
            c_dec = int(c, base=2)
            reconstructed_image[x_dec, y_dec] = c_dec
        
    return reconstructed_image.astype(np.uint8)
