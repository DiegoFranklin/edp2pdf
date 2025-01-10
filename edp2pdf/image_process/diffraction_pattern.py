import numpy as np
from typing import Tuple, Union


class eDiffractionPattern:
    """
    A class representing a diffraction pattern with associated data, center coordinates, and a mask.

    Attributes:
        data (np.ndarray): The diffraction pattern data as a 2D numpy array.
        center (Tuple[int, int]): The center coordinates of the diffraction pattern.
        mask (np.ndarray): The binary mask associated with the diffraction pattern.
    """

    def __init__(self, data: np.ndarray, center: Tuple[int, int], mask: np.ndarray):
        """
        Initializes the eDiffractionPattern object.

        Args:
            data (np.ndarray): The diffraction pattern data as a 2D numpy array.
            center (Tuple[int, int]): The center coordinates of the diffraction pattern.
            mask (np.ndarray): The binary mask associated with the diffraction pattern.

        Raises:
            TypeError: If `data` or `mask` are not numpy arrays, or if `center` is not a tuple of integers.
            ValueError: If `data` or `mask` are not 2D arrays, or if `center` is out of bounds.
        """
        # Validate input types
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if not isinstance(mask, np.ndarray):
            raise TypeError("Input `mask` must be a numpy array.")
        if not isinstance(center, tuple) or len(center) != 2 or not all(isinstance(c, Union[int, float]) for c in center):
            raise TypeError("Input `center` must be a tuple of two integers.")

        # Validate input shapes
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")
        if len(mask.shape) != 2:
            raise ValueError("Input `mask` must be a 2D array.")
        if data.shape != mask.shape:
            raise ValueError("Shapes of `data` and `mask` must match.")

        # Validate center coordinates
        if center[0] < 0 or center[1] < 0 or center[0] >= data.shape[0] or center[1] >= data.shape[1]:
            raise ValueError("Center coordinates are out of bounds.")

        self.data = data
        self.center = tuple(map(int, center))  # Ensure center coordinates are integers
        self._mask = mask

    @property
    def mask(self) -> np.ndarray:
        """
        Gets the binary mask associated with the diffraction pattern.

        Returns:
            np.ndarray: The binary mask as a 2D numpy array.
        """
        return self._mask

    @mask.setter
    def mask(self, mask_data: np.ndarray):
        """
        Sets the binary mask associated with the diffraction pattern.

        Args:
            mask_data (np.ndarray): The new binary mask as a 2D numpy array.

        Raises:
            TypeError: If `mask_data` is not a numpy array.
            ValueError: If `mask_data` is not a 2D array or does not match the shape of the data.
        """
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("Input `mask_data` must be a numpy array.")
        if len(mask_data.shape) != 2:
            raise ValueError("Input `mask_data` must be a 2D array.")
        if mask_data.shape != self.data.shape:
            raise ValueError("Shape of `mask_data` must match the shape of the data.")

        self._mask = mask_data