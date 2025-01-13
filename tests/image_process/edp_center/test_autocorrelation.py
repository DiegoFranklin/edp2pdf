import pytest
import numpy as np
from edp2pdf.image_process.edp_center.first_estimation.autocorrelation import autocorrelation
import cv2

def create_dummy_data(shape=(100, 100), sigma=10):
    """
    Creates a 2D Gaussian distribution centered in the middle of the array.

    Args:
        shape (tuple): Shape of the output array (rows, columns).
        sigma (float): Standard deviation of the Gaussian distribution.

    Returns:
        np.ndarray: A 2D array representing a centered Gaussian distribution.
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    # Create a grid of (x, y) coordinates
    x = np.arange(cols) - center_col
    y = np.arange(rows) - center_row
    x, y = np.meshgrid(x, y)

    # Compute the 2D Gaussian distribution
    gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return gaussian


def test_autocorrelation():
    image = create_dummy_data()
    center = autocorrelation(image, mask=np.ones_like(image))

    # Test return type
    assert isinstance(center, tuple)

    # Test centroid coordinates are within bounds
    assert 0 <= center[0] < image.shape[1]
    assert 0 <= center[1] < image.shape[0]
