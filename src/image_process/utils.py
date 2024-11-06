from typing import Tuple, Optional
import numpy as np

def get_centered_crop_img(image: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
    """
    Crops the given image to a centered square region based on the specified center.

    This function computes the maximum possible radius that maintains the cropped
    region within the image boundaries, and returns the cropped square portion
    centered at the given coordinates.

    Parameters
    ----------
    image : np.ndarray
        The input image to be cropped.
    center : Tuple[float, float]
        The (y, x) coordinates of the center point around which to crop the image.

    Returns
    -------
    np.ndarray
        The cropped square region of the input image.
    """
    c_x, c_y = map(int, center)
    radius = max_radius(image.shape, center)
    
    return image[c_x-radius:c_x+radius, c_y-radius:c_y+radius]

def euclidean_distance(a, b):
    """
    Calculates the Euclidean distance between two points a and b.

    Parameters
    ----------
    a : array_like
        The first point.
    b : array_like
        The second point.

    Returns
    -------
    float
        The Euclidean distance between a and b.
    """
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
    def __init__(self, data: np.ndarray, center: Tuple[int, int], max_radius: Optional[int] = None, mode: str = "linear_ramp") -> None:
        """
        Initializes the ImagePadder with the given data, center, and optional parameters.

        Parameters
        ----------
        data : np.ndarray
            The image data to be padded.
        center : Tuple[int, int]
            The center point for the padding operations.
        max_radius : Optional[int], default=None
            The maximum radius for padding. If None, it will be computed based on data.
        mode : str, default="linear_ramp"
            The mode of padding to be applied.
        """
        self._data: np.ndarray = data
        self._center: Tuple[int, int] = center
        self._mode: str = mode
        self._max_radius: Optional[int] = max_radius
        self._vertical_pad: Optional[Tuple[int, int]] = None 
        self._horizontal_pad: Optional[Tuple[int, int]] = None
        self._square_data: Optional[np.ndarray] = None

    def _compute_max_radius(self):
        if self._max_radius is None:
            self._max_radius = max_radius(self._data.shape, self._center)

    def _compute_pad_widths(self):
        """
        Computes the padding widths required on all four sides to center the image.

        The calculated padding widths are the maximum distances from the center to the
        edges of the image on each side, limited to the maximum radius. The padding
        widths are computed and stored in the instance variables self._vertical_pad
        and self._horizontal_pad.

        Modifies:
        self._vertical_pad (Tuple[int, int]): The padding widths for the vertical
            direction (up, down).
        self._horizontal_pad (Tuple[int, int]): The padding widths for the horizontal
            direction (left, right).
        """
        up = int(max(0, self._max_radius - self._center[0]))
        down = int(max(0, self._max_radius - (self._data.shape[0] - self._center[0] - 1)))
        left = int(max(0, self._max_radius - self._center[1]))
        right = int(max(0, self._max_radius - (self._data.shape[1] - self._center[1] - 1)))
        
        self._vertical_pad = (up, down)
        self._horizontal_pad = (left, right)

    def _compute_square_data(self):
        """
        Computes and stores the square padded data.

        This method first computes the maximum radius and the padding widths
        required to create a square image centered on the desired point. It then
        applies symmetric padding to create a square image centered on the desired
        point, using the specified mode. The computed padded data is stored in
        the instance variable self._square_data.

        Modifies:
        self._square_data (np.ndarray): The padded data, centered on the desired
            point and with the specified maximum radius.
        """

        self._compute_max_radius()
        self._compute_pad_widths()

        self._square_data = np.pad(
            self._data,
            pad_width=(self._vertical_pad, self._horizontal_pad),
            mode=self._mode
        )

    @property
    def square_padded_data(self) -> np.ndarray:
        """
        Returns the padded data, centered on the desired point and with the specified maximum radius.

        The padded data is computed and stored in the instance variable self._square_data
        the first time this property is accessed.

        Returns:
            np.ndarray: The padded data, centered on the desired point and with the specified
                maximum radius.
        """
        if self._square_data is None:
            self._compute_square_data()
        return self._square_data


    def recover_original_shape(self, data: np.ndarray) -> np.ndarray:
        """
        Recovers the original shape of the data from the padded data.

        Parameters
        ----------
        data : np.ndarray
            The padded data.

        Returns
        -------
        np.ndarray
            The original data, with the same shape as the input data to the
            constructor.
        """

        original_data: np.ndarray = data[
            self._vertical_pad[0]:data.shape[0] - self._vertical_pad[1], 
            self._horizontal_pad[0]:data.shape[1] - self._horizontal_pad[1]
        ]
        return original_data
