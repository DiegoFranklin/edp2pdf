from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List, Optional
from scipy.ndimage import gaussian_filter

import edp2pdf.image_process.mask.mask_ops as mo


class MaskGetter(ABC):
    """
    Abstract base class for generating binary masks from input data.
    """

    def __init__(self):
        """
        Initializes the MaskGetter.
        """
        pass

    @abstractmethod
    def get_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method to generate a binary mask based on the input data.

        Args:
            data (np.ndarray): The input data used to generate the mask.

        Returns:
            np.ndarray: A binary mask as a numpy array.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class RecursiveMask(MaskGetter):
    """
    RecursiveMask class that generates a binary mask through recursive thresholding
    and morphological operations.

    This class iteratively reduces a dynamic threshold (mean of the data) and applies
    morphological operations (dilation and erosion) to refine the mask, stopping when
    a defined ratio of the data is masked.
    """

    def __init__(self):
        """
        Initializes the RecursiveMask.
        """
        super().__init__()

    def get_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Generates a binary mask by recursively applying a decreasing threshold
        (mean of data) and refining the mask through morphological operations (dilate, erode).

        The process stops when the ratio of masked pixels to total pixels reaches 25%.

        Args:
            data (np.ndarray): The input data used to generate the mask.

        Returns:
            np.ndarray: The final binary mask after recursive thresholding and morphological operations.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is empty or has invalid dimensions.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if data.size == 0:
            raise ValueError("Input `data` must not be empty.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        op_data = data.copy()
        mask = np.zeros(op_data.shape, dtype=np.uint8)
        mask_superposition = mask.copy()
        mean_data = np.mean(op_data)

        ratio = 0

        while ratio < 0.25:
            op_data = np.logical_not(mask) * op_data
            mask = (op_data >= mean_data).astype(np.uint8)
            mean_data -= mean_data / 100

            dilate = mo.Dilate(iterations=2, kernel_size=2)
            erode = mo.Erode(iterations=20, kernel_size=2)
            routine = [dilate, erode]
            ocroutine = mo.OCRoutine(routine=routine)

            mask = ocroutine.execute_routine(mask)
            mask_superposition = np.logical_or(mask_superposition, mask)
            ratio = np.sum(mask_superposition) / np.prod(mask_superposition.shape)

        dilate = mo.Dilate(iterations=5, kernel_size=5)
        erode = mo.Erode(iterations=15, kernel_size=2)
        routine = [dilate, erode]
        ocroutine = mo.OCRoutine(routine=routine)

        mask_superposition = ocroutine.execute_routine(mask_superposition.astype(np.uint8))

        return mask_superposition


class GaussianBlurTreshMask(MaskGetter):
    """
    GaussianBlurTreshMask class that generates a binary mask using Gaussian blur and thresholding.
    """

    def __init__(self, sigma: int = 20, iterations: int = 10):
        """
        Initializes the GaussianBlurTreshMask with the given parameters.

        Args:
            sigma (int): Sigma for Gaussian blur. Defaults to 20.
            iterations (int): Number of iterations for Gaussian blur. Defaults to 10.

        Raises:
            ValueError: If `sigma` or `iterations` are not positive.
        """
        super().__init__()
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        if iterations <= 0:
            raise ValueError("Iterations must be positive.")
        self.sigma = sigma
        self.iterations = iterations

    def get_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Generates a binary mask using Gaussian blur and thresholding.

        Args:
            data (np.ndarray): The input data used to generate the mask.

        Returns:
            np.ndarray: A binary mask as a numpy array.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is empty or has invalid dimensions.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if data.size == 0:
            raise ValueError("Input `data` must not be empty.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        # Apply Gaussian filter iteratively
        limiter = data
        for _ in range(self.iterations):
            limiter = gaussian_filter(limiter, sigma=self.sigma)

        # Normalize and threshold
        limiter = np.sum(data) * limiter / np.sum(limiter)
        mask = data >= limiter

        return mask.astype(np.uint8)


class AdaptiveTreshMask(MaskGetter):
    """
    AdaptiveTreshMask class that generates a binary mask using adaptive thresholding.
    """

    def __init__(self, sector_size: int = 101, c_param: int = 0):
        """
        Initializes the AdaptiveTreshMask with the given parameters.

        Args:
            sector_size (int): Sector size for adaptive thresholding. Defaults to 101.
            c_param (int): Constant subtracted from the mean in adaptive thresholding. Defaults to 0.

        Raises:
            ValueError: If `sector_size` is not positive or odd.
        """
        super().__init__()
        if sector_size <= 0 or sector_size % 2 == 0:
            raise ValueError("Sector size must be a positive odd integer.")
        self.sector_size = sector_size
        self.c_param = c_param

    def get_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Generates a binary mask using adaptive thresholding.

        Args:
            data (np.ndarray): The input data used to generate the mask.

        Returns:
            np.ndarray: A binary mask as a numpy array.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is empty or has invalid dimensions.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if data.size == 0:
            raise ValueError("Input `data` must not be empty.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        # Normalize the data
        normalized_data = 255 * data / np.max(data)
        normalized_data = normalized_data.astype("uint8")

        # Apply adaptive thresholding
        mask = cv2.adaptiveThreshold(
            normalized_data,
            1,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            self.sector_size,
            self.c_param,
        )

        return mask.astype("uint8")


class MeanTreshMask(MaskGetter):
    """
    MeanTreshMask class that generates a binary mask based on the mean of the data.
    """

    def __init__(self, constant: float = 0.2):
        """
        Initializes the MeanTreshMask with the given constant.

        Args:
            constant (float): Constant value to multiply the standard deviation subtracted from the mean. Defaults to 0.2.

        Raises:
            ValueError: If `constant` is negative.
        """
        super().__init__()
        if constant < 0:
            raise ValueError("Constant must be non-negative.")
        self.constant = constant

    def get_mask(self, data: np.ndarray) -> np.ndarray:
        """
        Generates a binary mask based on the mean of the data, with an optional constant adjustment.

        Args:
            data (np.ndarray): The input data used to generate the mask.

        Returns:
            np.ndarray: A binary mask as a numpy array.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is empty or has invalid dimensions.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if data.size == 0:
            raise ValueError("Input `data` must not be empty.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        # Calculate the mean of the data
        mean_value = np.mean(data)

        # Create the mask based on mean minus constant
        mask = data >= (mean_value - self.constant * np.std(data))

        return mask.astype(np.uint8)


def superpose_masks(data: np.ndarray, mask_list: List[MaskGetter]) -> np.ndarray:
    """
    Superimposes multiple masks using a logical OR operation.

    Args:
        data (np.ndarray): The input data used to generate the masks.
        mask_list (List[MaskGetter]): List of MaskGetter instances from which to obtain masks.

    Returns:
        np.ndarray: A binary mask as a numpy array, representing the superposition of all masks.

    Raises:
        ValueError: If `mask_list` is empty.
        TypeError: If `data` is not a numpy array or `mask_list` contains invalid objects.
    """
    if not mask_list:
        raise ValueError("The `mask_list` cannot be empty.")
    if not isinstance(data, np.ndarray):
        raise TypeError("Input `data` must be a numpy array.")
    if not all(isinstance(mask_getter, MaskGetter) for mask_getter in mask_list):
        raise TypeError("All elements in `mask_list` must be instances of MaskGetter.")

    # Initialize the superposition with the first mask
    superposition = mask_list[0].get_mask(data)

    # Combine masks using logical OR
    for mask_getter in mask_list[1:]:
        superposition = np.logical_or(superposition, mask_getter.get_mask(data))

    return superposition.astype(np.uint8)