# geqie/encodings/naqss/map.py
import numpy as np
from qiskit.quantum_info import Operator


def _rotation(theta: float) -> np.ndarray:
    """
    Real 2x2 rotation matrix used in the NAQSS paper for Rx / Ry
    (they are just planar rotations):

        R(θ) = [[cos θ, -sin θ],
                [sin θ,  cos θ]]
    """
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=complex)


def map(*coords: int, R: int, image: np.ndarray) -> Operator:
    """
    NAQSS map δ(coords, p_coords) for arbitrary-dimensional images.

    - coords: index tuple into `image`, e.g.
        * 2D: (u, v)
        * 3D: (x, y, z)
        * etc.
    - Color encoded as a rotation by φ in [0, π/2] (FRQI-style) on a 'color' qubit.
    - Segmentation encoded as β ∈ {0, π/2} on a separate 'segmentation' qubit.

    Current simple 2-class segmentation:
        * foreground  (label 1) if pixel >= global mean
        * background (label 0) otherwise

    The full map operator acts on 2 qubits:
        R_seg(β) ⊗ R_col(φ)
    so its dimension is 4x4.
    """

    # --- grayscale intensity & color angle φ (FRQI-like) ---
    # coords is a tuple, so image[coords] works for any ndim
    pixel_value = float(image[coords])
    phi = pixel_value / 255.0 * (np.pi / 2.0)  # φ ∈ [0, π/2]

    R_color = _rotation(phi)

    # --- segmentation: β ∈ {0, π/2} from F2 for m = 2 sub-images ---
    # background / foreground split by global mean intensity
    mean_intensity = float(image.mean())
    segment_label = 1 if pixel_value >= mean_intensity else 0  # 0=background, 1=target

    beta = 0.0 if segment_label == 0 else np.pi / 2.0
    R_seg = _rotation(beta)

    # tensor product: segmentation (more significant) ⊗ color (less significant)
    map_operator = np.kron(R_seg, R_color)

    return Operator(map_operator)
