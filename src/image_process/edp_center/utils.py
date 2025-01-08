import numpy as np
from typing import Tuple, List


def get_centered_crop_img(image: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
    """
    Crops the image to the largest possible area centered on the specified center point.

    Args:
        image (np.ndarray): The input image as a 2D numpy array.
        center (Tuple[float, float]): The (x, y) coordinates of the center point.

    Returns:
        np.ndarray: The cropped image centered on the specified point.

    Raises:
        ValueError: If the center is outside the image boundaries or if the image is not 2D.
        TypeError: If the input is not a numpy array or if the center is not a tuple of floats.
    """
    # Validate input types
    if not isinstance(image, np.ndarray):
        raise TypeError("Input `image` must be a numpy array.")
    if not isinstance(center, tuple) or len(center) != 2 or not all(isinstance(c, (int, float)) for c in center):
        raise TypeError("Input `center` must be a tuple of two floats.")

    # Ensure the image is 2D
    if len(image.shape) != 2:
        raise ValueError("Input `image` must be a 2D array.")

    # Round the center coordinates to the nearest integer
    c_x, c_y = map(round, center[::-1])

    # Get image dimensions
    x_length, y_length = image.shape

    # Check if the center is within the image boundaries
    if c_x < 0 or c_x >= x_length or c_y < 0 or c_y >= y_length:
        raise ValueError("Center coordinates are outside the image boundaries.")

    # Calculate the maximum possible radius for cropping
    radius = min(c_x, c_y, abs(x_length - c_x), abs(y_length - c_y))

    # Perform the cropping
    return image[c_x - radius:c_x + radius, c_y - radius:c_y + radius]


def bilinear_interpolation(
    point: Tuple[float, float],
    neighbors: Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]],
    values: Tuple[float, float, float, float]
) -> float:
    """
    Performs bilinear interpolation for a given point based on the values of its four neighbors.

    Args:
        point (Tuple[float, float]): A tuple (x, y) representing the coordinates where interpolation is desired.
        neighbors (Tuple[Tuple[int, int], ...]): A tuple containing the four corner integer coordinates:
                                                 ((x0, y0), (x0, y1), (x1, y0), (x1, y1)).
        values (Tuple[float, float, float, float]): A tuple containing the values at the four corners:
                                                   (value_00, value_01, value_10, value_11).

    Returns:
        float: The interpolated value at the specified point.

    Raises:
        ValueError: If the point or neighbors are invalid or if the values are not provided for all four neighbors.
        TypeError: If the inputs are not of the expected types.
    """
    # Validate input types
    if not isinstance(point, tuple) or len(point) != 2 or not all(isinstance(c, (int, float)) for c in point):
        raise TypeError("Input `point` must be a tuple of two floats.")
    if not isinstance(neighbors, tuple) or len(neighbors) != 4 or not all(isinstance(n, tuple) and len(n) == 2 for n in neighbors):
        raise TypeError("Input `neighbors` must be a tuple of four coordinate tuples.")

    # Extract coordinates and values
    x, y = point
    (x0, y0), (x0, y1), (x1, y0), (x1, y1) = neighbors
    value_00, value_01, value_10, value_11 = values

    # Perform bilinear interpolation
    Q11 = value_00 * (x1 - x) * (y1 - y)
    Q12 = value_01 * (x1 - x) * (y - y0)
    Q21 = value_10 * (x - x0) * (y1 - y)
    Q22 = value_11 * (x - x0) * (y - y0)

    interpolated_value = (Q11 + Q12 + Q21 + Q22) / ((x1 - x0) * (y1 - y0))

    return interpolated_value


def get_integer_neighbors(point: Tuple[float, float]) -> Tuple[Tuple[int, int], ...]:
    """
    Finds the four integer neighbors of a given point for bilinear interpolation.

    Args:
        point (Tuple[float, float]): A tuple (x, y) representing the coordinates of the point.

    Returns:
        Tuple[Tuple[int, int], ...]: A tuple containing the four integer neighbors:
                                     [(x_floor, y_floor), (x_floor, y_ceil),
                                      (x_ceil, y_floor), (x_ceil, y_ceil)].

    Raises:
        TypeError: If the input is not a tuple of two floats.
    """
    # Validate input type
    if not isinstance(point, tuple) or len(point) != 2 or not all(isinstance(c, (int, float)) for c in point):
        raise TypeError("Input `point` must be a tuple of two floats.")

    # Calculate floor and ceiling coordinates
    i_floor, j_floor = int(np.floor(point[0])), int(np.floor(point[1]))
    i_ceil, j_ceil = i_floor + 1, j_floor + 1

    # Create the list of neighbors
    neighbors = [
        (i_floor, j_floor),
        (i_floor, j_ceil),
        (i_ceil, j_floor),
        (i_ceil, j_ceil)
    ]

    return tuple(neighbors)