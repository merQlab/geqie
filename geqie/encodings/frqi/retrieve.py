import numpy as np
import json
from typing import Union, Dict

def retrieve(results: Union[str, Dict]) -> np.ndarray:
    """
    Decodes an image from quantum state measurement results using FRQI logic.

    Parameters:
    results (str | dict): A JSON string or dictionary where keys are binary 
                          strings representing quantum states, and values are 
                          their respective occurrence counts.

    Returns:
    np.ndarray: A NumPy array representing the decoded image.
    """
    # -------------------------------------------------------------------------
    # SECURITY FIX: Parse string input using json.loads(), NEVER eval()
    # -------------------------------------------------------------------------
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except json.JSONDecodeError as e:
            raise ValueError(f"Security: Input is not valid JSON. {e}")
    
    # Validation to prevent crashes on empty results
    if not results:
        # Return an empty 1x1 array or raise error depending on preference
        print("Warning: Retrieval results are empty.")
        return np.zeros((1, 1))

    # -------------------------------------------------------------------------
    # FRQI Logic (Flexible Representation of Quantum Images)
    # -------------------------------------------------------------------------
    # Get length of the first key (e.g., "00001" -> 5 qubits)
    state_length = len(next(iter(results)))
    
    # FRQI typically uses 1 qubit for color information
    color_qubits = 1
    number_of_position_qubits = state_length - color_qubits
    
    # Calculate dimensions (assuming square/rectangular split)
    x_qubits = number_of_position_qubits // 2
    y_qubits = number_of_position_qubits // 2
    
    image_shape = (2**x_qubits, 2**y_qubits)
    reconstructed_image = np.zeros(image_shape)
    ones = np.zeros_like(reconstructed_image)
    total = np.zeros_like(reconstructed_image)

    for state, n in results.items():
        if not isinstance(state, str):
            continue

        # Position is the first part, Color is the last qubit
        b = state[:-1]
        c = state[-1]

        # Convert binary position to integer index
        try:
            m = int(b, base=2)
        except ValueError:
            continue # Skip invalid binary strings

        # Bounds check to prevent index out of bounds errors
        if m < total.size:
            total.flat[m] += n
            if c == "1":
                ones.flat[m] += n

    # Reconstruct: Pixel value is (Count of 1s) / (Total Counts)
    # Using np.where to handle division by zero safely
    try:
        reconstructed_image = np.where(total > 0, ones / total, 0)
    except Exception as e:
        print(f"Error during FRQI image retrieval: {e}")
        
    return reconstructed_image