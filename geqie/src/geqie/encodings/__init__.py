"""Common utilities for quantum image encodings."""

import numpy as np


def ensure_grayscale(image: np.ndarray, strategy: str | None = None) -> np.ndarray:
    """
    Convert image to 2D for encodings that expect grayscale images.
    
    Args:
        image: Input image array (2D or 3D)
        strategy: Grayscale conversion strategy
            - "luminance" (default): Use RGB luminance formula `(0.299*R + 0.587*G + 0.114*B)`
            - "first_channel": Take first channel
            - "average": Average all channels
    
    Returns:
        2D image array
    
    Raises:
        ValueError: If image is not 2D or 3D, or strategy is unknown
    """
    strategy = strategy or "luminance"

    if image.ndim == 2:
        return image

    if image.ndim == 3:
        if strategy == "first_channel":
            return image[:, :, 0]
        elif strategy == "average":
            return image.mean(axis=2).astype(image.dtype)
        elif strategy == "luminance":
            # Standard RGB to grayscale conversion
            weights = np.array([0.299, 0.587, 0.114])
            return (image[:,:,:3] @ weights).astype(image.dtype)
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'first_channel', 'average', or 'luminance'.")
    
    raise ValueError(f"Expected 2D single- or multi-channel or image, received shape: '{image.shape}'")
