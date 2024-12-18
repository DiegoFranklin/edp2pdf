from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from functools import lru_cache
import numpy as np
import concurrent.futures

from src.image_process.edp_center.utils import get_centered_crop_img, bilinear_interpolation, get_integer_neighbors
from src.image_process.edp_center.center_optimization.distance_metrics import masked_metric_factory

class OptFunc(ABC):
    """Abstract base class for optimization functions."""
    
    def __init__(self, data: np.ndarray, mask: Optional[np.ndarray] = None, distance_metric: str = 'manhattan'):

        self._data = data
        self._mask = mask if mask is not None else np.ones_like(data)
        self._distance_metric = distance_metric

    def get_penalty_func(self) -> Callable[[Tuple[float, float]], float]:
        """Returns the penalty function for point evaluation."""
        return self._get_float_point_evaluation
    
    def _get_float_point_evaluation(self, point: Tuple[float, float]) -> float:
        """Calculates the penalty for a given float point.

        If the point has non-integer coordinates, bilinear interpolation is used.
        Otherwise, it evaluates the point directly.
        """
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
        """Evaluates a specific integer point and returns a float value."""
        pass



class Distance(OptFunc):
    def __init__(self, data: np.ndarray, mask: np.ndarray, distance_metric: str = 'manhattan'):
        super().__init__(data, mask, distance_metric)

    @lru_cache(maxsize=128)
    def _get_point_evaluation(self, point: Tuple[int, int]) -> float:
        i, j = point

        cropped_data = get_centered_crop_img(self._data, (i, j))
        cropped_mask = get_centered_crop_img(self._mask, (i, j))

        flipped_data = np.flip(cropped_data)
        flipped_mask = np.flip(cropped_mask)

        distance_function = masked_metric_factory(self._distance_metric)

        distance = distance_function(cropped_data, flipped_data, np.logical_and(cropped_mask, flipped_mask))
        return distance
    
