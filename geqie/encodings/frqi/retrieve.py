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
    color_qubits = 1
    number_of_position_qubits = state_length - color_qubits
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2
    
    image_shape = (2**x_qubits, 2**y_qubits)

    reconstructed_image = np.zeros((image_shape[0], image_shape[1]))

    ones = np.zeros_like(reconstructed_image)
    total = np.zeros_like(reconstructed_image)

    for state, n in results.items():
        b = state[:-1]
        c = state[-1]

        m = int(b, base=2)
        total.flat[m] += n
        if c == "1":
            ones.flat[m] += n
    try:
        reconstructed_image = np.where(total > 0, np.arccos(np.sqrt(1 - ones / total)), 0)
        reconstructed_image = 255 * 2 * reconstructed_image / np.pi
        reconstructed_image = reconstructed_image.astype(np.uint8)
    except ZeroDivisionError:
        print("Error during FRQI image retrieval. Division by zero!")
        
    return reconstructed_image
