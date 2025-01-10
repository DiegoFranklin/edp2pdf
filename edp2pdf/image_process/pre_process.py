from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from scipy import signal, ndimage


class PreProcess(ABC):
    """
    Abstract base class for preprocessing operations.

    Subclasses should implement the `pre_process` method to perform specific
    preprocessing operations on input data.
    """

    @abstractmethod
    def pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method to preprocess the input data.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The preprocessed data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class AllPositive(PreProcess):
    """
    A preprocessing class that shifts the data to ensure all values are positive.
    """

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Shifts the data so that the minimum value is zero.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The preprocessed data with all values non-negative.

        Raises:
            TypeError: If `data` is not a numpy array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        return data - np.min(data)


class MedianFilter(PreProcess):
    """
    A preprocessing class that applies a median filter to the input data.
    """

    def __init__(self, kernel_size: int = 3):
        """
        Initializes the MedianFilter with the specified kernel size.

        Args:
            kernel_size (int): The size of the median filter kernel. Defaults to 3.

        Raises:
            ValueError: If `kernel_size` is not a positive odd integer.
        """
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be a positive odd integer.")
        self._kernel_size = kernel_size

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Applies a median filter to the input data.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The median-filtered data.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")
        return signal.medfilt2d(data, self._kernel_size)


class GaussianBlur(PreProcess):
    """
    A preprocessing class that applies a Gaussian blur to the input data.
    """

    def __init__(self, sigma: float = 3):
        """
        Initializes the GaussianBlur with the specified sigma value.

        Args:
            sigma (float): The standard deviation of the Gaussian kernel. Defaults to 3.

        Raises:
            ValueError: If `sigma` is not a positive number.
        """
        if sigma <= 0:
            raise ValueError("Sigma must be a positive number.")
        self._sigma = sigma

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Applies a Gaussian blur to the input data.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The Gaussian-blurred data.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")
        return ndimage.gaussian_filter(data, self._sigma)


class Resizer(PreProcess):
    """
    A preprocessing class that resizes the input data using interpolation.
    """

    def __init__(self, scaling_factor: float):
        """
        Initializes the Resizer with the specified scaling factor.

        Args:
            scaling_factor (float): The scaling factor for resizing the data.

        Raises:
            ValueError: If `scaling_factor` is not a positive number.
        """
        if scaling_factor <= 0:
            raise ValueError("Scaling factor must be a positive number.")
        self.scaling_factor = scaling_factor

    def pre_process(self, data: np.ndarray) -> np.ndarray:
        """
        Resizes the input data using interpolation.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The resized data.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        if self.scaling_factor > 1:
            order = 3  # Cubic interpolation for upscaling
        elif self.scaling_factor < 1:
            order = 1  # Linear interpolation for downscaling
        else:
            return data  # No resizing needed

        resized_data = ndimage.zoom(data, self.scaling_factor, order=order)
        return resized_data


class PreProcessPipe:
    """
    A class for chaining multiple preprocessing operations into a pipeline.
    """

    def __init__(self, pre_processors: Optional[List[PreProcess]] = None):
        """
        Initializes the PreProcessPipe with an optional list of preprocessors.

        Args:
            pre_processors (Optional[List[PreProcess]]): A list of preprocessing operations. Defaults to an empty list.

        Raises:
            TypeError: If `pre_processors` contains objects that are not instances of PreProcess.
        """
        if pre_processors is not None and not all(isinstance(p, PreProcess) for p in pre_processors):
            raise TypeError("All elements in `pre_processors` must be instances of PreProcess.")
        self._pre_processors = pre_processors or []

    def add_pre_processor(self, pre_processor: PreProcess):
        """
        Adds a preprocessing operation to the pipeline.

        Args:
            pre_processor (PreProcess): The preprocessing operation to add.

        Raises:
            TypeError: If `pre_processor` is not an instance of PreProcess.
        """
        if not isinstance(pre_processor, PreProcess):
            raise TypeError("Input `pre_processor` must be an instance of PreProcess.")
        self._pre_processors.append(pre_processor)

    def pre_process_pipe(self, data: np.ndarray) -> np.ndarray:
        """
        Applies all preprocessing operations in the pipeline to the input data.

        Args:
            data (np.ndarray): The input data to preprocess.

        Returns:
            np.ndarray: The preprocessed data after applying all operations.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        for pre_processor in self._pre_processors:
            data = pre_processor.pre_process(data)
        return data