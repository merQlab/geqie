import json

import numpy as np


def retrieve(results: str) -> np.ndarray:
    """
    Decodes an image from quantum state measurement results.

    Parameters:
    results (dict): A dictionary where keys are binary strings representing quantum states,
                         and values are their respective occurrence counts.

    Returns:
    np.ndarray: A NumPy array representing the decoded image.
    """

    results = json.loads(results)

    state_length = len(next(iter(results)))
    row_col_shape = int((state_length - 1) / 2) ** 2
    image_shape = (row_col_shape, row_col_shape)

    image_reconstructed = np.zeros(image_shape, dtype=np.uint8)

    ones = np.zeros_like(image_reconstructed)
    total = np.zeros_like(image_reconstructed)

    for state, n in results.items():
        b = state[:-1]
        c = state[-1]

        m = int(b, base=2)
        total.flat[m] += n
        if c == "1":
            ones.flat[m] += n

    image_reconstructed = ones / total
    try:
        image_reconstructed = np.where(total > 0, ones / total, 0)
    except ZeroDivisionError:
        print("Error during FRQI image retrieval. Division by zero!")

    print(image_reconstructed)
    return image_reconstructed
