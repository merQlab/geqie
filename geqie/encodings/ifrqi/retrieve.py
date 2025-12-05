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
    color_qubits = 4
    number_of_position_qubits = state_length - color_qubits
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2
    
    image_shape = (2**x_qubits, 2**y_qubits)

    reconstructed_image = np.zeros((image_shape[0], image_shape[1]))

    reconstructed_image_counts_total = np.zeros((image_shape[0], image_shape[1]), dtype=int)
    reconstructed_image_counts_ones = np.zeros((image_shape[0], image_shape[1], color_qubits), dtype=int)

    for state, n in results.items():  
        x = state[0: x_qubits]
        y = state[y_qubits: number_of_position_qubits]

        c = state[number_of_position_qubits: number_of_position_qubits + color_qubits][::-1]

        x_dec = int(x, base=2)
        y_dec = int(y, base=2)

        reconstructed_image_counts_total[x_dec, y_dec] += n

        for count in range(len(c)):
            if c[count] == "1":
                reconstructed_image_counts_ones[x_dec, y_dec, count] += n
            
    for x in range(reconstructed_image_counts_ones.shape[0]):
        for y in range(reconstructed_image_counts_ones.shape[1]):
            color_bit_string = ""
            for c in range(color_qubits):
                coeff = reconstructed_image_counts_ones[x, y, c]/reconstructed_image_counts_total[x, y]
                if coeff > .75:
                    color_bit_string += "11"
                elif .5 < coeff < .75:
                    color_bit_string += "10"
                elif .25 < coeff < .5:
                    color_bit_string += "01"
                elif coeff < .25:
                    color_bit_string += "00"

            reconstructed_image[x, y] = int(color_bit_string[::-1], base=2)
        
    return reconstructed_image
