import numpy as np
from typing import Tuple

def get_centroid(image: np.ndarray) -> Tuple[float, float]:
    """
    Calculates the centroid of a grayscale image using pixel intensity as mass.

    Args:
        image (np.ndarray): A 2D array representing the grayscale image.

    Returns:
        Tuple[float, float]: The (x, y) coordinates of the centroid.

    Raises:
        ValueError: If the input is not a 2D array or if the total mass (sum of pixel intensities) is zero.
        TypeError: If the input is not a numpy array.
    """
    # Validate input type
    if not isinstance(image, np.ndarray):
        raise TypeError("Input `image` must be a numpy array.")

    # Ensure the input is a 2D array
    if len(image.shape) != 2:
        raise ValueError("Input must be a 2D array representing a grayscale image.")

    # Get the dimensions of the image
    height, width = image.shape

    # Create coordinate grids
    x_coords = np.arange(width)
    y_coords = np.arange(height)

    # Compute the total mass (sum of pixel intensities)
    total_mass = np.sum(image)

    # Check if the total mass is zero
    if total_mass == 0:
        raise ValueError("Total mass (sum of pixel intensities) is zero, cannot compute centroid.")

    # Calculate the weighted coordinates
    x_centroid = np.sum(image * x_coords[None, :]) / total_mass
    y_centroid = np.sum(image * y_coords[:, None]) / total_mass

    return x_centroid, y_centroid