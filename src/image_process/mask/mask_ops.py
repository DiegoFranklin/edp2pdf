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

        :param mask_data: Input binary mask data as a numpy array.
        :return: The result of the operation as a numpy array.
        """
        pass

class Dilate(MaskOperation):
    """
    Class to perform dilation on a binary mask.

    Dilation is used to expand the boundaries of foreground objects in a binary mask.
    """
    
    def __init__(self, iterations: int = 5, kernel_size: int = 5):
        """
        Initializes the Dilate operation with the given parameters.

        :param iterations: Number of times the dilation operation is applied.
        :param kernel_size: Size of the structuring element (disk) used for dilation.
        """
        self.iterations = iterations
        self._kernel = disk(kernel_size)

    def operate(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Applies dilation to the input mask data.

        :param mask_data: Input binary mask data as a numpy array.
        :return: The dilated mask data as a numpy array.
        """
        return cv2.dilate(mask_data, kernel=self._kernel, iterations=self.iterations)

class Erode(MaskOperation):
    """
    Class to perform erosion on a binary mask.

    Erosion is used to shrink the boundaries of foreground objects in a binary mask.
    """
    
    def __init__(self, iterations: int = 5, kernel_size: int = 5):
        """
        Initializes the Erode operation with the given parameters.

        :param iterations: Number of times the erosion operation is applied.
        :param kernel_size: Size of the structuring element (disk) used for erosion.
        """
        self.iterations = iterations
        self._kernel = disk(kernel_size)

    def operate(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Applies erosion to the input mask data.

        :param mask_data: Input binary mask data as a numpy array.
        :return: The eroded mask data as a numpy array.
        """
        return cv2.erode(mask_data, kernel=self._kernel, iterations=self.iterations)

class OCRoutine:
    """
    Class to execute a sequence of mask operations.

    The routine applies a series of mask operations in the order they are provided.
    """
    
    def __init__(self, routine: Optional[List[MaskOperation]] = None):
        """
        Initializes the OCRoutine with a list of mask operations.

        :param routine: List of MaskOperation instances to be executed in sequence.
                        If None, initializes with an empty list.
        """
        self._routine = routine or []

    def execute_routine(self, mask_data: np.ndarray) -> np.ndarray:
        """
        Executes the sequence of mask operations on the input mask data.

        :param mask_data: Input binary mask data as a numpy array.
        :return: The mask data after applying all operations in the routine.
        """
        for operation in self._routine:
            mask_data = operation.operate(mask_data)
        return mask_data
