from abc import abstractmethod, ABC
import cv2
import numpy as np
from typing import Tuple

from ..utils import max_radius


class PolarTransformation(ABC):
    """
    Abstract base class for polar transformation operations.

    Subclasses should implement the `transform` method to perform specific
    polar transformations on input data.
    """

    @abstractmethod
    def transform(self, data: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
        """
        Abstract method to perform a polar transformation on the input data.

        Args:
            data (np.ndarray): The input data to transform.
            center (Tuple[float, float]): The center point for the polar transformation.

        Returns:
            np.ndarray: The transformed polar image.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class CVPolarTransformation(PolarTransformation):
    """
    A class for performing polar transformations using OpenCV's `warpPolar` function.

    Attributes:
        _interpolation_method (int): The interpolation method used for the transformation.
        _max_radius (float): The maximum radius for the polar transformation.
        _polar_image_size (Tuple[int, int]): The size of the output polar image.
    """

    def __init__(self, interpolation_method: int = cv2.INTER_CUBIC):
        """
        Initializes the CVPolarTransformation with the given interpolation method.

        Args:
            interpolation_method (int): The interpolation method to use. Defaults to cv2.INTER_CUBIC.

        Raises:
            ValueError: If the interpolation method is invalid.
        """
        if interpolation_method not in {
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        }:
            raise ValueError("Invalid interpolation method.")
        self._interpolation_method = interpolation_method
        self._max_radius: float = None
        self._polar_image_size: Tuple[int, int] = None

    def _compute_max_radius(self, data_shape: Tuple[int, int], center: Tuple[int, int]) -> None:
        """
        Computes the maximum radius for the polar transformation.

        Args:
            data_shape (Tuple[int, int]): The shape of the input data.
            center (Tuple[int, int]): The center point for the polar transformation.

        Raises:
            ValueError: If the center is invalid or out of bounds.
        """
        if self._max_radius is None:
            self._max_radius = max_radius(data_shape, center)

    def _compute_polar_image_size(self, data_shape: Tuple[int, int], center: Tuple[int, int]) -> None:
        """
        Computes the size of the output polar image.

        Args:
            data_shape (Tuple[int, int]): The shape of the input data.
            center (Tuple[int, int]): The center point for the polar transformation.

        Raises:
            ValueError: If the center is invalid or out of bounds.
        """
        if self._polar_image_size is None:
            self._compute_max_radius(data_shape, center)
            radial_size = round(np.sqrt(np.pi) * self._max_radius)
            azimuth_size = round(np.sqrt(np.pi) * self._max_radius)
            self._polar_image_size = (radial_size, azimuth_size)

    def _check_input_center(self, data_shape: Tuple[int, int], center: Tuple[int, int]) -> None:
        """
        Validates the center point for the polar transformation.

        Args:
            data_shape (Tuple[int, int]): The shape of the input data.
            center (Tuple[int, int]): The center point for the polar transformation.

        Raises:
            IndexError: If the center is out of bounds.
        """
        if (
            center[0] < 0
            or center[1] < 0
            or center[0] >= data_shape[0]
            or center[1] >= data_shape[1]
        ):
            raise IndexError("Center index out of bounds.")

    def transform(self, data: np.ndarray, center: Tuple[float, float]) -> np.ndarray:
        """
        Transforms the input data into polar coordinates using OpenCV's `warpPolar`.

        Args:
            data (np.ndarray): The input data to transform.
            center (Tuple[float, float]): The center point for the polar transformation.

        Returns:
            np.ndarray: The transformed polar image.

        Raises:
            TypeError: If `data` is not a numpy array or `center` is not a tuple.
            ValueError: If `data` is empty or has invalid dimensions.
            IndexError: If the center is out of bounds.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if data.size == 0:
            raise ValueError("Input `data` must not be empty.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")
        if not isinstance(center, tuple) or len(center) != 2:
            raise TypeError("Input `center` must be a tuple of two floats.")

        self._check_input_center(data.shape, center)
        self._compute_max_radius(data.shape, center)
        self._compute_polar_image_size(data.shape, center)

        polar_image = cv2.warpPolar(
            data,
            self._polar_image_size,
            center,
            self._max_radius,
            self._interpolation_method,
        )

        return polar_image