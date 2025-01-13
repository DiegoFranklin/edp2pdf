import numpy as np
import cv2
from skimage.morphology import disk
from typing import Optional, List
from abc import ABC, abstractmethod


class MaskOperation(ABC):
    """
    Abstract base class for mask operations.

    Subclasses should implement the `operate` method to perform specific
    operations on mask data.
    """

    @abstractmethod
    def operate(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform an operation on the given mask data.

        Args:
            mask_data (np.ndarray): Input binary mask data as a numpy array.

        Returns:
            np.ndarray: The result of the operation as a numpy array.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class Dilate(MaskOperation):
    """
    Class to perform dilation on a binary mask.

    Dilation is used to expand the boundaries of foreground objects in a binary mask.
    """

    def __init__(self, iterations: int = 5, kernel_size: int = 5):
        """
        Initializes the Dilate operation with the given parameters.

        Args:
            iterations (int): Number of times the dilation operation is applied. Defaults to 5.
            kernel_size (int): Size of the structuring element (disk) used for dilation. Defaults to 5.

        Raises:
            ValueError: If `iterations` or `kernel_size` are not positive.
        """
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")
        if kernel_size <= 0:
            raise ValueError("Kernel size must be positive.")

        self.iterations = iterations
        self._kernel = disk(kernel_size)

    def operate(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Applies dilation to the input mask data.

        Args:
            mask_data (np.ndarray): Input binary mask data as a numpy array.

        Returns:
            np.ndarray: The dilated mask data as a numpy array.

        Raises:
            TypeError: If `mask_data` is not a numpy array.
            ValueError: If `mask_data` is not a binary mask or has invalid dimensions.
        """
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("Input `mask_data` must be a numpy array.")
        if len(mask_data.shape) != 2:
            raise ValueError("Input `mask_data` must be a 2D array.")

        return cv2.dilate(mask_data, kernel=self._kernel, iterations=self.iterations)


class Erode(MaskOperation):
    """
    Class to perform erosion on a binary mask.

    Erosion is used to shrink the boundaries of foreground objects in a binary mask.
    """

    def __init__(self, iterations: int = 5, kernel_size: int = 5):
        """
        Initializes the Erode operation with the given parameters.

        Args:
            iterations (int): Number of times the erosion operation is applied. Defaults to 5.
            kernel_size (int): Size of the structuring element (disk) used for erosion. Defaults to 5.

        Raises:
            ValueError: If `iterations` or `kernel_size` are not positive.
        """
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")
        if kernel_size <= 0:
            raise ValueError("Kernel size must be positive.")

        self.iterations = iterations
        self._kernel = disk(kernel_size)

    def operate(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Applies erosion to the input mask data.

        Args:
            mask_data (np.ndarray): Input binary mask data as a numpy array.

        Returns:
            np.ndarray: The eroded mask data as a numpy array.

        Raises:
            TypeError: If `mask_data` is not a numpy array.
            ValueError: If `mask_data` is not a binary mask or has invalid dimensions.
        """
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("Input `mask_data` must be a numpy array.")
        if len(mask_data.shape) != 2:
            raise ValueError("Input `mask_data` must be a 2D array.")

        return cv2.erode(mask_data, kernel=self._kernel, iterations=self.iterations)


class OCRoutine:
    """
    Class to execute a sequence of mask operations.

    The routine applies a series of mask operations in the order they are provided.
    """

    def __init__(self, routine: Optional[List[MaskOperation]] = None):
        """
        Initializes the OCRoutine with a list of mask operations.

        Args:
            routine (Optional[List[MaskOperation]]): List of MaskOperation instances to be executed in sequence.
                                                    If None, initializes with an empty list.

        Raises:
            TypeError: If `routine` contains objects that are not instances of MaskOperation.
        """
        if routine is not None and not all(isinstance(op, MaskOperation) for op in routine):
            raise TypeError("All elements in `routine` must be instances of MaskOperation.")
        self._routine = routine or []

    def execute_routine(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Executes the sequence of mask operations on the input mask data.

        Args:
            mask_data (np.ndarray): Input binary mask data as a numpy array.

        Returns:
            np.ndarray: The mask data after applying all operations in the routine.

        Raises:
            TypeError: If `mask_data` is not a numpy array.
            ValueError: If `mask_data` is not a binary mask or has invalid dimensions.
        """
        if not isinstance(mask_data, np.ndarray):
            raise TypeError("Input `mask_data` must be a numpy array.")
        if len(mask_data.shape) != 2:
            raise ValueError("Input `mask_data` must be a 2D array.")

        for operation in self._routine:
            mask_data = operation.operate(mask_data)
        return mask_data