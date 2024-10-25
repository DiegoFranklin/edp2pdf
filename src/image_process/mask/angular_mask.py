import numpy as np
from abc import ABC, abstractmethod

from src.image_process.mask.mask_getters import MeanTreshMask
from src.image_process.mask.mask_ops import Dilate, Erode, OCRoutine

class IAngularMask(ABC):
    def __init__(self, angular_range_expansion: float = None,
                       cyclic_shift: int = None,
                       num_radius_beams: int = 20,
                       ocroutine: OCRoutine = None):

        self._ocroutine = ocroutine or OCRoutine([Erode(iterations=5, kernel_size=2),
                                                  Dilate(iterations=5, kernel_size=2)])
        self._num_radius_beams = num_radius_beams
        self._angular_range_expansion = angular_range_expansion
        self._cyclic_shift = cyclic_shift
    
    @abstractmethod
    def get_angular_mask(self, polar_representation):
        pass

class MeanAngularMask(IAngularMask):
    def __init__(self, angular_range_expansion: float = None,
                       cyclic_shift: int = None,
                       num_radius_beams: int = 20,
                       ocroutine: OCRoutine = None):
        super().__init__(angular_range_expansion=angular_range_expansion, 
                         cyclic_shift=cyclic_shift, 
                         num_radius_beams=num_radius_beams, 
                         ocroutine=ocroutine)
    
    def get_angular_mask(self, polar_representation):

        data_section = polar_representation._full_polar_image[:,
                polar_representation._start_radial_index:polar_representation._start_radial_index+self._num_radius_beams]
        
        mask_getter = MeanTreshMask(constant=0)

        mask_section = mask_getter.get_mask(data_section)
        mask_section = self._ocroutine.execute_routine(mask_section)

        angular_mask = mask_section[:, 0]

        if self._angular_range_expansion is not None:
            angular_mask = self._expand_hole(angular_mask, polar_representation._full_theta_space)
        if self._cyclic_shift is not None:
            angular_mask = self._compute_cyclic_shift(angular_mask, polar_representation._full_theta_space)

        return angular_mask

    def _expand_hole(self, precursor_mask: np.array, theta: np.array) -> np.array:
        num_beams = round(self._angular_range_expansion*theta.shape[0]/360)

        expanded_hole_mask = precursor_mask.copy()
        for i in range(-num_beams, num_beams):
            rolled_mask = np.roll(precursor_mask, shift=i)
            expanded_hole_mask = np.logical_and(expanded_hole_mask, rolled_mask)

        return expanded_hole_mask.copy()
    
    def _compute_cyclic_shift(self, precursor_mask: np.array, theta: np.array) -> np.array:
        shift = np.argmin(np.abs((self._cyclic_shift % 360) - theta))
        return np.logical_and(precursor_mask, np.roll(precursor_mask, shift))

        


