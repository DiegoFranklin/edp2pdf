from abc import abstractmethod, ABC
import cv2
import numpy as np
from typing import Tuple

class PolarTransformation(ABC):
    @abstractmethod
    def transform(self, data: np.ndarray, center: Tuple[float, float]):
        pass

class CVPolarTransformation(PolarTransformation):

    def __init__(self, interpolation_method : int = cv2.INTER_CUBIC):

        self._interpolation_method  = interpolation_method

    def transform(self,
                  data: np.ndarray,
                  center: Tuple[float, float]
                    ) -> np.ndarray:

        out_bounds_error = IndexError('Center index out of bounds.')
        if center[0] > data.shape[0] or center[1] > data.shape[1]:
            raise out_bounds_error
        if center[0] < 0 or center[1] < 0:
            raise IndexError('Center index out of bounds.')


        def euclidean_distance(p1, p2):
            """Calculate the Euclidean distance between two points."""
            return np.linalg.norm(np.array(p1) - np.array(p2))

        # Define the corners of the image
        corners = [
            (0, 0),  # Top-left
            (0, data.shape[1] - 1),  # Top-right
            (data.shape[0] - 1, 0),  # Bottom-left
            (data.shape[0] - 1, data.shape[1] - 1)  # Bottom-right
        ]

        # Calculate the maximum distance from the center to the corners
        max_radius = round(np.max([euclidean_distance(center, corner) for corner in corners]))

        radial_size = max_radius
        azimuth_size = round(np.multiply(*data.shape)/radial_size)


        polar_image = cv2.warpPolar(data,
                                    (azimuth_size, radial_size),
                                    center,
                                    max_radius,
                                    self._interpolation_method )

        return polar_image