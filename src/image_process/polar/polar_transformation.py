from abc import abstractmethod, ABC
import cv2
import numpy as np
from typing import Tuple

from ..utils import distance

class PolarTransformation(ABC):
    @abstractmethod
    def transform(self, data: np.ndarray, center: Tuple[float, float]):
        pass

class CVPolarTransformation(PolarTransformation):

    def __init__(self, interpolation_method : int = cv2.INTER_CUBIC):

        self._interpolation_method  = interpolation_method
        self._max_radius: float = None
        self._polar_image_size: Tuple[int, int] = None
    
    def _compute_max_radius(self, data_shape: Tuple[int, int], center: Tuple[int, int]) -> None:

        if self._max_radius is None:
            corners = [
                (0, 0),  
                (0, data_shape[1] - 1),  
                (data_shape[0] - 1, 0), 
                (data_shape[0] - 1, data_shape[1] - 1)  
            ]

            # Calculate the maximum distance from the center to the corners
            self._max_radius = np.max([distance(center, corner) for corner in corners])

    def _compute_polar_image_size(self, data_shape: Tuple[int, int], center: Tuple[int, int]) -> None:
        if self._polar_image_size is None:

            self._compute_max_radius(data_shape, center)
            radial_size = round(np.sqrt(np.pi) * self._max_radius)

            azimuth_size = round(np.sqrt(np.pi) * self._max_radius)

            self._polar_image_size = (radial_size, azimuth_size)

    def _check_input_center(self, data_shape: Tuple[int, int], center: Tuple[int, int]) -> None:
        
        out_bounds_error = IndexError('Center index out of bounds.')
        if center[0] > data_shape[0] or center[1] > data_shape[1]:
            raise out_bounds_error
        if center[0] < 0 or center[1] < 0:
            raise out_bounds_error
    
    
    def transform(self,
                  data: np.ndarray,
                  center: Tuple[float, float]
                    ) -> np.ndarray:

        self._check_input_center(data.shape, center)

        self._compute_max_radius(data.shape, center)

        self._compute_polar_image_size(data.shape, center)


        polar_image = cv2.warpPolar(data,
                                    self._polar_image_size,
                                    center,
                                    self._max_radius,
                                    self._interpolation_method )

        return polar_image