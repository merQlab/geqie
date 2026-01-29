"""Common utilities for quantum image encodings."""

import numpy as np


def ensure_2d(image: np.ndarray, strategy: str = "first_channel") -> np.ndarray:
    """
    Convert 3D image to 2D for encodings that expect grayscale images.
    
    Args:
        image: Input image array (2D or 3D)
        strategy: Conversion strategy for 3D images:
            - "first_channel": Take first channel (default)
            - "average": Average all channels
            - "luminance": Use RGB luminance formula (0.299*R + 0.587*G + 0.114*B)
    
    Returns:
        2D image array
    
    Raises:
        ValueError: If image is not 2D or 3D, or strategy is unknown
    """
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
            return (image @ weights).astype(image.dtype)
        else:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Use 'first_channel', 'average', or 'luminance'."
            )
    
    raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")
