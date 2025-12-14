import numpy as np
import json

def retrieve(results: str, image_dimensionality: int = 2) -> np.ndarray:
    state_length = len(next(iter(results)))
    color_qubits = 1
    number_of_position_qubits = state_length - color_qubits

    dimension_qubits = number_of_position_qubits // image_dimensionality
    
    image_shape = [2**dimension_qubits] * image_dimensionality

    reconstructed_image = np.zeros(image_shape)

    ones = np.zeros_like(reconstructed_image)
    total = np.zeros_like(reconstructed_image)

    for state, n in results.items():
        b = state[:-1]
        c = state[-1]

        m = int(b, base=2)
        total.flat[m] += n
        if c == "1":
            ones.flat[m] += n
    
    reconstructed_image = np.where(total > 0, np.arccos(np.sqrt(1 - ones / total)), 0)
    reconstructed_image = 255 * 2 * reconstructed_image / np.pi
    reconstructed_image = reconstructed_image.astype(np.uint8)
    return reconstructed_image
