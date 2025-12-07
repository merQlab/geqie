# geqie/encodings/naqss/data.py
import numpy as np
from qiskit.quantum_info import Statevector


def data(*coords: int, R: int, image: np.ndarray) -> Statevector:
    """
    Position encoding |ξ(coords)⟩ for an arbitrary-dimensional array.

    Parameters
    ----------
    coords
        Integer index tuple into `image`, e.g.
        * 2D: (u, v)
        * 3D: (x, y, z)
        * etc.
    R
        Number of qubits per spatial dimension.
    image
        Intensity array. Its first len(coords) axes are treated as spatial.

    Returns
    -------
    Statevector
        Basis state |index⟩ in a register of size 2**(len(coords) * R).
    """
    n_dims = len(coords)
    if n_dims == 0:
        raise ValueError("At least one coordinate is required")

    spatial_shape = image.shape[:n_dims]
    n_pos_qubits = n_dims * R
    dim = 2 ** n_pos_qubits

    # Check coords within bounds
    for i, (c, size) in enumerate(zip(coords, spatial_shape)):
        if c < 0 or c >= size:
            raise ValueError(
                f"Coordinate {i} = {c} out of bounds for axis size {size}"
            )

    # Check that R is large enough for the spatial size
    n_points = int(np.prod(spatial_shape))
    if n_points > dim:
        raise ValueError(
            f"R={R} and {n_dims} dims give position space {dim}, "
            f"but image has {n_points} points. Increase R."
        )

    # Row-major flattening (matches NumPy default, and encode side):
    #   index = np.ravel_multi_index(coords, spatial_shape, order="C")
    index = np.ravel_multi_index(coords, spatial_shape, order="C")

    # Extra safety: index must fit the allocated Hilbert space
    if index >= dim:
        raise ValueError(f"Flattened index {index} exceeds position register size {dim}")

    data_vector = np.zeros(dim, dtype=int)
    data_vector[index] = 1
    return Statevector(data_vector)
