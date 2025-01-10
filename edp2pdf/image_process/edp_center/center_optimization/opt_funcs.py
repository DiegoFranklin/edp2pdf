from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from functools import lru_cache
import numpy as np
import concurrent.futures

from edp2pdf.image_process.edp_center.utils import get_centered_crop_img, bilinear_interpolation, get_integer_neighbors
from edp2pdf.image_process.edp_center.center_optimization.distance_metrics import masked_metric_factory


class OptFunc(ABC):
    """
    Abstract base class for optimization functions.

    Attributes:
        _data (np.ndarray): Input data array for optimization.
        _mask (np.ndarray): Binary mask applied to the data. Defaults to an array of ones if not provided.
        _distance_metric (str): Name of the distance metric to use. Defaults to 'manhattan'.
    """

    def __init__(self, data: np.ndarray, mask: Optional[np.ndarray] = None, distance_metric: str = 'manhattan'):
        """
        Initializes the OptFunc class.

        Args:
            data (np.ndarray): Input data array for optimization.
            mask (Optional[np.ndarray]): Binary mask applied to the data. Defaults to an array of ones if not provided.
            distance_metric (str): Name of the distance metric to use. Defaults to 'manhattan'.

        Raises:
            ValueError: If `data` and `mask` have incompatible shapes.
            TypeError: If `data` or `mask` are not numpy arrays.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if mask is not None and not isinstance(mask, np.ndarray):
            raise TypeError("Input `mask` must be a numpy array.")
        if mask is not None and data.shape != mask.shape:
            raise ValueError("Shapes of `data` and `mask` must match.")

        self._data = data
        self._mask = mask if mask is not None else np.ones_like(data)
        self._distance_metric = distance_metric

    def get_penalty_func(self) -> Callable[[Tuple[float, float]], float]:
        """
        Returns the penalty function for evaluating a point.

        Returns:
            Callable[[Tuple[float, float]], float]: A function that calculates the penalty for a given point.
        """
        return self._get_float_point_evaluation

    def _get_float_point_evaluation(self, point: Tuple[float, float]) -> float:
        """
        Calculates the penalty for a given point with float coordinates.

        If the point has non-integer coordinates, bilinear interpolation is used.
        Otherwise, it evaluates the point directly.

        Args:
            point (Tuple[float, float]): The point to evaluate, represented as (x, y) coordinates.

        Returns:
            float: The calculated penalty value.

        Raises:
            ValueError: If the point coordinates are invalid (e.g., negative or out of bounds).
        """
        point = tuple(point)
        if not all(isinstance(coord, (int, float)) for coord in point):
            raise ValueError("Point coordinates must be numeric.")
        if point[0] < 0 or point[1] < 0:
            raise ValueError("Point coordinates must be non-negative.")

        # Check for integer coordinates
        if not point[0].is_integer() or not point[1].is_integer():
            neighbors = get_integer_neighbors(point)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                area_diffs = list(executor.map(self._get_point_evaluation, neighbors))

            return bilinear_interpolation(point, neighbors, area_diffs)
        else:
            return self._get_point_evaluation(tuple(map(int, point)))

    @abstractmethod
    @lru_cache(maxsize=128)
    def _get_point_evaluation(self) -> Callable[[Tuple[int]], float]:
        """
        Abstract method to evaluate a specific integer point and return a float value.

        Returns:
            Callable[[Tuple[int]], float]: A function that evaluates a point with integer coordinates.
        """
        pass


class Distance(OptFunc):
    """
    A class for calculating distance-based penalties using a specified metric.

    Inherits from `OptFunc` and implements the `_get_point_evaluation` method.
    """

    def __init__(self, data: np.ndarray, mask: np.ndarray, distance_metric: str = 'manhattan'):
        """
        Initializes the Distance class.

        Args:
            data (np.ndarray): Input data array for optimization.
            mask (np.ndarray): Binary mask applied to the data.
            distance_metric (str): Name of the distance metric to use. Defaults to 'manhattan'.

        Raises:
            ValueError: If `data` and `mask` have incompatible shapes.
            TypeError: If `data` or `mask` are not numpy arrays.
        """
        super().__init__(data, mask, distance_metric)

    @lru_cache(maxsize=128)
    def _get_point_evaluation(self, point: Tuple[int, int]) -> float:
        """
        Evaluates a specific integer point and returns the distance-based penalty.

        Args:
            point (Tuple[int, int]): The point to evaluate, represented as (x, y) integer coordinates.

        Returns:
            float: The calculated distance-based penalty.

        Raises:
            ValueError: If the point coordinates are invalid (e.g., negative or out of bounds).
        """
        if not all(isinstance(coord, int) for coord in point):
            raise ValueError("Point coordinates must be integers.")
        if point[0] < 0 or point[1] < 0:
            raise ValueError("Point coordinates must be non-negative.")

        i, j = point

        cropped_data = get_centered_crop_img(self._data, (i, j))
        cropped_mask = get_centered_crop_img(self._mask, (i, j))

        flipped_data = np.flip(cropped_data)
        flipped_mask = np.flip(cropped_mask)

        distance_function = masked_metric_factory(self._distance_metric)

        distance = distance_function(cropped_data, flipped_data, np.logical_and(cropped_mask, flipped_mask))
        return distance