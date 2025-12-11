# geqie/encodings/naqss/retrieve.py
import json
from typing import Dict, Iterable, Tuple, Union

import numpy as np


def _ensure_counts(result: Union[str, Dict[str, int]]) -> Dict[str, int]:
    """
    Accept either a JSON string (as produced by `geqie simulate`)
    or a dict already parsed.
    """
    if isinstance(result, str):
        result = json.loads(result)
    if not isinstance(result, dict):
        raise TypeError("Result must be a JSON string or a dict of counts")
    return {k: int(v) for k, v in result.items()}


def retrieve(
    result: Union[str, Dict[str, int]],
    shape: Union[None, Iterable[int], Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Retrieve only the reconstructed image from NAQSS measurement results.

    Parameters
    ----------
    result
        JSON string or dict of counts: {"bitstring": shots, ...}, as returned by `geqie simulate --encoding naqss`.
    shape
        Optional spatial shape used in encoding, e.g.
            (H, W)      for 2D
            (X, Y, Z)   for 3D
        If omitted, we assume the original 2D square case:
            total_pos_qubits = 2R, side = 2**R, shape = (side, side)

    Returns
    -------
    np.ndarray
        nD uint8 array with reconstructed intensities.
    """
    counts = _ensure_counts(result)
    if not counts:
        raise ValueError("Empty result: no counts to retrieve from")

    # --- basic qubit bookkeeping ---
    sample_key = next(iter(counts))
    n_qubits = len(sample_key)
    if n_qubits < 3:
        raise ValueError(f"Expected at least 3 qubits (position + seg + color), got {n_qubits}")

    # 2 map qubits: segmentation + color
    n_pos_qubits = n_qubits - 2

    # --- deduce / validate spatial shape ---
    if shape is None:
        # Backwards-compatible 2D square assumption: total_pos_qubits = 2R
        if n_pos_qubits % 2 != 0:
            raise ValueError(
                "Without an explicit `shape`, NAQSS retrieve assumes 2D square "
                f"images with total_pos_qubits = 2R. Got n_pos_qubits = {n_pos_qubits}."
            )
        R = n_pos_qubits // 2
        side = 2 ** R
        spatial_shape = (side, side)
    else:
        spatial_shape = tuple(int(s) for s in shape)
        if any(s <= 0 for s in spatial_shape):
            raise ValueError(f"Invalid spatial shape {spatial_shape}")
        n_points = int(np.prod(spatial_shape))
        pos_dim = 2 ** n_pos_qubits
        if n_points > pos_dim:
            raise ValueError(
                f"Shape {spatial_shape} has {n_points} points but position register supports only {pos_dim} states. "
                "Check R and/or encoding settings."
            )

    n_points = int(np.prod(spatial_shape))

    # --- accumulators over spatial points ---
    total_per_pixel = np.zeros(spatial_shape, dtype=int)
    color1_per_pixel = np.zeros_like(total_per_pixel)

    for bitstring, c in counts.items():
        if len(bitstring) != n_qubits:
            raise ValueError("Inconsistent bitstring length in counts")

        # bitstring is big-endian: [q_{n-1} ... q_0]
        pos_bits = bitstring[:n_pos_qubits]
        # segmentation qubit = bitstring[n_pos_qubits] (ignored here)
        color_bit = bitstring[n_pos_qubits + 1]  # color qubit

        index = int(pos_bits, 2)
        if index >= n_points:
            # index corresponds to padding outside the actual image volume
            continue

        coords = np.unravel_index(index, spatial_shape, order="C")

        total_per_pixel[coords] += c
        if color_bit == "1":
            color1_per_pixel[coords] += c

    # --- reconstruct intensities (FRQI-like inversion) ---
    image = np.zeros(spatial_shape, dtype=float)

    for coords in np.ndindex(spatial_shape):
        tot = total_per_pixel[coords]
        if tot == 0:
            continue

        p_color1 = color1_per_pixel[coords] / tot

        # p_color1 ≈ sin^2(φ), φ ∈ [0, π/2]
        # φ = arcsin(sqrt(p_color1))
        # I ≈ 255 * (2/π) * φ
        phi_est = np.arcsin(np.sqrt(np.clip(p_color1, 0.0, 1.0)))
        image[coords] = phi_est * (2.0 / np.pi) * 255.0

    image_uint8 = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return image_uint8
