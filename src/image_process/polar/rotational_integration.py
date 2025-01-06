import numpy as np

from .polar_representation import PolarRepresentation
from src.image_process.diffraction_pattern import eDiffractionPattern

class RotationalIntegration:

    def __init__(self, polar_representation: PolarRepresentation):
        self._polar_representation = polar_representation
        self.extended_polar_mask = None

    def _construct_extended_polar_mask(self) -> np.ndarray:

        mask_edp = eDiffractionPattern(data=self._polar_representation.edp.mask,
                                       center=self._polar_representation.edp.center,
                                       mask=self._polar_representation.edp.mask)
        
        mask_polar = PolarRepresentation(mask_edp)
        mask_polar.radial_range = self._polar_representation.radial_range
        
        full_edp = eDiffractionPattern(data=np.ones_like(self._polar_representation.edp.mask),
                                       center=self._polar_representation.edp.center,
                                       mask=np.ones_like(self._polar_representation.edp.mask))
        
        full_polar = PolarRepresentation(full_edp)
        full_polar.radial_range = self._polar_representation.radial_range

        polar_mask = mask_polar.polar_image.copy()
        lock = np.argmax(np.sum(polar_mask, axis=0))
        lock = round(lock * 0.9)

        last_angular_mask = polar_mask[:,lock - 1]
        for i in range(0, round(lock * 0.05)):
            last_angular_mask = np.logical_or(last_angular_mask, polar_mask[:,lock - 1 - i])

        full_angular_mask = np.tile(last_angular_mask, (full_polar.polar_image.shape[1], 1)).T

        polar_mask[:,lock:] = 0



        polar_full = full_polar.polar_image.copy()
        polar_full = np.logical_and(polar_full, full_angular_mask)
        polar_full[:,:lock] = 0

        final_mask = np.logical_or(polar_mask, polar_full)

        import matplotlib.pyplot as plt
        plt.imshow(final_mask)
        plt.show()

        return final_mask

    def get_masked_rotational_integration(self,
                                          start_angle: float,
                                          end_angle: float,
                                          method: str = 'mean') -> np.ndarray:

        if self.extended_polar_mask is None:
            self.extended_polar_mask = self._construct_extended_polar_mask()
        
        self._polar_representation.angular_range = (start_angle, end_angle)

        polar_data = self._polar_representation.polar_image
        final_mask = self.extended_polar_mask
        integral = []
        for i in range(polar_data.shape[1]):
            col = polar_data[:,i]
            col_mask = final_mask[:,i]

            if method == 'median':
                integral.append(np.median(col[col_mask == 1]))
            elif method == 'mean':
                integral.append(np.mean(col[col_mask == 1]))

        return np.array(integral)

    def get_rotational_integration(self, start_angle: float, end_angle: float, method: str = 'mean') -> np.ndarray:
        self._polar_representation.angular_range = (start_angle, end_angle)
        if method == 'mean':
            return np.mean(self._polar_representation.polar_image, axis=0)
        if method == 'median':
            return np.median(self._polar_representation.polar_image, axis=0)
        else:
            raise ValueError(f'Unknown method: {method}')