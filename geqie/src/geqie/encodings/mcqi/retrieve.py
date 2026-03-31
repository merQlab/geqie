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
    number_of_position_qubits = state_length - 3
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2

    image_shape = (2**x_qubits, 2**y_qubits)

    reconstructed_image = np.zeros((image_shape[0], image_shape[1], 3))

    ones = np.zeros((image_shape[0], image_shape[1], 3))
    total = np.zeros((image_shape[0], image_shape[1], 3))

    for state, n in results.items():
        if n > 0:
            x = state[0:x_qubits]
            y = state[x_qubits: 2*y_qubits]
            x_dec = int(x, base=2)
            y_dec = int(y, base=2)
            c = state[2*y_qubits: 2*y_qubits+3]

            # if Red:
            if c[0:2] == "00":
                if c[-1] == "1":
                    ones[x_dec, y_dec, 0] = n
                total[x_dec, y_dec, 0] += n
            # if Green:
            if c[0:2] == "01":
                if c[-1] == "1":
                    ones[x_dec, y_dec, 1] = n
                total[x_dec, y_dec, 1] += n
            # if Blue:
            if c[0:2] == "10":
                if c[-1] == "1":
                    ones[x_dec, y_dec, 2] = n
                total[x_dec, y_dec, 2] += n
            
    reconstructed_image = np.where(total > 0, np.arccos(np.sqrt(1 - ones / total)), 0)
    reconstructed_image = 255 * 2 * reconstructed_image / np.pi
    reconstructed_image = reconstructed_image.astype(np.uint8)    
    return reconstructed_image