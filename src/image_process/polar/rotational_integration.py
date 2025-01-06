import numpy as np

from .polar_representation import PolarRepresentation

class RotationalIntegration:

    def __init__(self, polar_representation: PolarRepresentation):
        self._polar_representation = polar_representation

    def get_rotational_integration(self, start_angle: float, end_angle: float, mode: str = 'mean') -> np.ndarray:
        self._polar_representation.angular_range = (start_angle, end_angle)
        if mode == 'mean':
            return np.mean(self._polar_representation.polar_image, axis=0)
        if mode == 'median':
            return np.median(self._polar_representation.polar_image, axis=0)
        else:
            raise ValueError(f'Unknown mode: {mode}')
