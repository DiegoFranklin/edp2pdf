from abc import ABC, abstractmethod
import cv2
import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter

import src.image_process.mask.mask_ops as mo

class MaskGetter(ABC):
    def __init__(self):
        """
        Initializes the MaskGetter with the given data.

        """

    @abstractmethod
    def get_mask(self, data) -> np.ndarray:
        """
        Abstract method to generate a mask based on given parameters.

        :return: A binary mask as a numpy array.
        """
        pass

class RecursiveMask(MaskGetter):

    """
    RecursiveMask class that generates a binary mask through recursive thresholding
    and morphological operations.
    
    This class iteratively reduces a dynamic threshold (mean of the data) and applies
    morphological operations (dilation and erosion) to refine the mask, stopping when
    a defined ratio of the data is masked.
    
    Attributes:
    -----------
    _data : np.ndarray
        Input data used for mask generation. Inherited from MaskGetter.
        
    Methods:
    --------
    __init__(data: np.ndarray):
        Initializes the RecursiveMask with the given data.
        
    get_mask() -> np.ndarray:
        Generates the mask by applying thresholding and morphological operations.
        Returns the final binary mask as a NumPy array.
    """

    def __init__(self):
        super().__init__()

    def get_mask(self, data: np.array) -> np.ndarray:

        """
        Generates a binary mask by recursively applying a decreasing threshold
        (mean of data) and refining the mask through morphological operations (dilate, erode).
        
        The process stops when the ratio of masked pixels to total pixels reaches 25%.
        
        Returns:
        --------
        np.ndarray:
            The final binary mask after recursive thresholding and morphological operations.
        """
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

class GaussianBlurTreshgMask(MaskGetter):
    def __init__(self, sigma: int = 20, iterations: int = 10):
        """
        Initializes the GaussianBlurTreshgMask with the given data and parameters.

        :param sigma: Sigma for Gaussian blur.
        :param iterations: Number of iterations for Gaussian blur.
        """
        super().__init__()
        self.sigma = sigma
        self.iterations = iterations

    def get_mask(self, data: np.array) -> np.ndarray:
        """
        Generates a binary mask using Gaussian blur and thresholding.

        :return: A binary mask as a numpy array.
        """
        # Apply Gaussian filter iteratively
        limiter = data
        for _ in range(self.iterations):
            limiter = gaussian_filter(limiter, sigma=self.sigma)

        # Normalize and threshold
        limiter = np.sum(data) * limiter / np.sum(limiter)
        mask = data >= limiter

        return mask.astype(np.uint8)

class AdaptiveTreshMask(MaskGetter):
    def __init__(self, sector_size: int = 101, c_param: int = 0):
        """
        Initializes the AdaptiveTreshMask with the given data and parameters.

        :param sector_size: Sector size for adaptive thresholding.
        :param c_param: Constant subtracted from the mean in adaptive thresholding.
        """
        super().__init__()
        self.sector_size = sector_size
        self.c_param = c_param

    def get_mask(self, data: np.array) -> np.ndarray:
        """
        Generates a binary mask using adaptive thresholding.

        :return: A binary mask as a numpy array.
        """
        # Normalize the data
        normalized_data = 255 * data / np.max(data)
        normalized_data = normalized_data.astype('uint8')

        # Apply adaptive thresholding
        mask = cv2.adaptiveThreshold(normalized_data, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, self.sector_size, self.c_param)

        return mask.astype('uint8')

class MeanTreshMask(MaskGetter):
    def __init__(self, constant: int = 0.2):
        """
        Initializes the MeanTreshMask with the given data and parameters.

        :param constant: Constant value to mmultiply the stdsubtract from the mean.
        """
        super().__init__()
        self.constant = constant

    def get_mask(self, data: np.array) -> np.ndarray:
        """
        Generates a binary mask based on the mean of the data, with an optional constant adjustment.

        :return: A binary mask as a numpy array.
        """
        # Calculate the mean of the data
        mean_value = np.mean(data)

        # Create the mask based on mean minus constant
        mask = data >= (mean_value - self.constant*np.std(data))

        return mask.astype(np.uint8)

    
def superpose_masks(data: np.array, mask_list: List[MaskGetter]) -> np.ndarray:
    """
    Superimposes multiple masks using a logical OR operation.
    
    :param data: input data
    :param mask_list: List of MaskGetter instances from which to obtain masks.
    :return: A binary mask as a numpy array, representing the superposition of all masks.
    """
    if not mask_list:
        raise ValueError("The mask_list cannot be empty.")

    # Initialize the superposition with the first mask
    superposition = mask_list[0].get_mask(data)

    # Combine masks using logical OR
    for mask_getter in mask_list[1:]:
        superposition = np.logical_or(superposition, mask_getter.get_mask(data))

    return superposition.astype(np.uint8)