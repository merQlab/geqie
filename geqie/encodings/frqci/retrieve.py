import numpy as np
import json

from .map import BLUE_SHIFT, NORMALIZATION_FACTOR, GREEN_SHIFT, RED_SHIFT

RETRIEVE_MASK_8BIT = np.uint32(0xFF)


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

    ones = np.zeros(image_shape)
    total = np.zeros(image_shape)

    for state, n in results.items():
        b = state[:-1]
        c = state[-1]

        m = int(b, base=2)
        total.flat[m] += n
        if c == "1":
            ones.flat[m] += n

    p1 = np.where(total > 0, ones / total, 0.0)

    theta = np.arcsin(np.sqrt(p1))
    raw_f = theta * (NORMALIZATION_FACTOR / (np.pi / 2))

    raw = np.rint(raw_f).astype(np.uint32)
    raw = np.clip(raw, 0, NORMALIZATION_FACTOR).astype(np.uint32)

    red   = (raw >> RED_SHIFT) & RETRIEVE_MASK_8BIT
    green = (raw >> GREEN_SHIFT) & RETRIEVE_MASK_8BIT
    blue  = (raw >> BLUE_SHIFT) & RETRIEVE_MASK_8BIT

    reconstructed_image = np.stack([red, green, blue], axis=-1)

    # reconstructed_image = 255 * reconstructed_image
    reconstructed_image = reconstructed_image.astype(np.uint8)
    return reconstructed_image
