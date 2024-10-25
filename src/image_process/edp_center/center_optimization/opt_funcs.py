from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional
from functools import lru_cache
import numpy as np
import concurrent.futures

from src.image_process.edp_center.utils import get_centered_crop_img, bilinear_interpolation, get_integer_neighbors
from src.image_process.diffraction_pattern import eDiffractionPattern
from src.image_process.polar.polar_representation import PolarRepresentation
from src.image_process.polar.rotational_average import RotationalAverage

class OptFunc(ABC):
    """Abstract base class for optimization functions."""
    
    def __init__(self, data: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Initializes the optimization function with data and an optional mask.

        Parameters:
        data (np.ndarray): The input data array.
        mask (Optional[np.ndarray]): An optional mask array. If not provided, a default mask is created.
        """
        self._data = data
        self._mask = mask if mask is not None else np.ones_like(data)

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



class Area(OptFunc):
    def __init__(self, data: np.ndarray, mask: np.ndarray):
        super().__init__(data, mask)

    @lru_cache(maxsize=128)
    def _get_point_evaluation(self, point: Tuple[int, int]) -> float:
        i, j = point

        cropped_data = get_centered_crop_img(self._data, (i, j))
        cropped_mask = get_centered_crop_img(self._mask, (i, j))

        flipped_data = cropped_data[::-1, ::-1]
        flipped_mask = cropped_mask[::-1, ::-1]

        diff_data = np.abs(cropped_data - flipped_data) * (cropped_mask & flipped_mask)

        return np.mean(diff_data)
    
class Distance(OptFunc):
    
    @lru_cache(maxsize=128)
    def _get_point_evaluation(self, point: Tuple[int, int]) -> float:

        edp = eDiffractionPattern(self._data, point)
        polar_representation = PolarRepresentation(edp, start=0.1, end=0.6)

        r = 10.0
        cyclic_shift=180
        polar_representation.set_angular_mask_getter_params(cyclic_shift=cyclic_shift, angular_range_expansion=r)
        
        rotational_average = RotationalAverage(polar_representation)

        probe_angle_pairs = [(angle, angle+180) for angle in np.arange(0, 180, 45)
                            if polar_representation.angular_mask[np.argmin(np.abs(angle-polar_representation.theta))]]

        rotational_averages = [(rotational_average.get_rotational_average(probe_pair[0] - r, probe_pair[0] + r),
                                rotational_average.get_rotational_average(probe_pair[1] - r, probe_pair[1] + r))
                                for probe_pair in probe_angle_pairs]
        
        correlation = [np.corrcoef(* rot_avg) for rot_avg in rotational_averages]

        return - np.mean(correlation)

    