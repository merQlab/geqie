import numpy as np

from qiskit.quantum_info import Operator

RED_SHIFT   = 16
GREEN_SHIFT = 8
BLUE_SHIFT  = 0
NORMALIZATION_FACTOR = 2**24 - 1

def map(u: int, v: int, R: int, image: np.ndarray) -> Operator:
    """FRQCI mapping function"""
    red   = image[u, v, 0].astype(np.uint32)
    green = image[u, v, 1].astype(np.uint32)
    blue  = image[u, v, 2].astype(np.uint32)

    color_blend = (red << RED_SHIFT) + (green << GREEN_SHIFT) + (blue << BLUE_SHIFT)

    theta = color_blend / NORMALIZATION_FACTOR * (np.pi / 2)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    map_operator = [
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta],
    ]

    return Operator(map_operator)

