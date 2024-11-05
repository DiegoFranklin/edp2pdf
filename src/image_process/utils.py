from typing import Tuple
import numpy as np

def get_centered_crop_img(image, center):
    # crops the image to biggest area possible so its get centered on center
    c_x = round(center[1])
    c_y = round(center[0])

    x_lenth = image.shape[0]
    y_lenth = image.shape[1]

    radius = min(c_x, c_y, abs(x_lenth-c_x), abs(y_lenth-c_y))

    return image[c_x-radius:c_x+radius, c_y-radius:c_y+radius]

def euclidean_distance(a, b):
    c = np.subtract(a, b)

    return np.linalg.norm(c)


def max_radius(data_shape: Tuple[int, int], center: Tuple[int, int]) -> int:
    """
    Calculates the maximum radius of a circle centered at the given point, that
    does not exceed the boundaries of the given data shape.

    Parameters
    ----------
    data_shape : Tuple[int, int]
        Shape of the data.
    center : Tuple[int, int]
        Center of the circle.

    Returns
    -------
    int
        The maximum radius of the circle.
    """
    corners = [
        (0, 0), 
        (0, data_shape[1] - 1),
        (data_shape[0] - 1, 0),
        (data_shape[0] - 1, data_shape[1] - 1) 
    ]
    # Calculate the maximum distance from the center to the corners
    max_radius = int(np.ceil(np.max([euclidean_distance(center, corner) for corner in corners])))

    return max_radius


class ImagePadder:
    def __init__(self, data, center, max_radius=None, mode="linear_ramp"):
        self._data = data
        self._center = center
        self._mode = mode
        self._max_radius = max_radius
        self._up = None
        self._down = None
        self._right = None
        self._left = None
        self._square_data = None

    def _compute_max_radius(self):
        if self._max_radius is None:
            self._max_radius = max_radius(self._data.shape, self._center)

    def _compute_pad_widths(self):
        self._up = max(0, self._max_radius - self._center[0])
        self._down = max(0, self._max_radius - (self._data.shape[0] - self._center[0] - 1))
        self._left = max(0, self._max_radius - self._center[1])
        self._right = max(0, self._max_radius - (self._data.shape[1] - self._center[1] - 1))

    def _compute_square_data(self):
        self._compute_max_radius()
        self._compute_pad_widths()

        self._square_data = np.pad(
            self._data,
            pad_width=(
                (round(self._up), round(self._down)),
                (round(self._left), round(self._right))
            ),
            mode=self._mode
        )

    @property
    def square_data(self):
        if self._square_data is None:
            self._compute_square_data()
        return self._square_data

    def recover_original_shape(self, data):
        original_data = data[
            round(self._up):round(data.shape[0] - self._down), 
            round(self._left):round(data.shape[1] - self._right)
        ]
        return original_data