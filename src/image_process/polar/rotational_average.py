import numpy as np

from .polar_representation import PolarRepresentation

class RotationalAverage:

    def __init__(self, polar_representation: PolarRepresentation):
        self._polar_representation = polar_representation

    def get_rotational_average(self, start_angle: float, end_angle: float) -> np.ndarray:
        self._polar_representation.angular_range = (start_angle, end_angle)
        return np.mean(self._polar_representation.polar_image, axis=0)
