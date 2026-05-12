import numpy as np
import sympy as sp

from qiskit.quantum_info import Operator


def map(u: int, v: int, R: int, image: np.ndarray, **_) -> Operator | np.ndarray:
    pixel_value = image[u, v]
    is_symbolic = isinstance(pixel_value, sp.Basic)

    if is_symbolic:
        theta = pixel_value / 255.0 * (sp.pi / 2)
        return np.array(
            [
                [sp.cos(theta), -sp.sin(theta)],
                [sp.sin(theta), sp.cos(theta)],
            ],
            dtype=object,
        )

    theta = float(pixel_value) / 255.0 * (np.pi / 2)
    map_operator = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )

    return Operator(map_operator)
