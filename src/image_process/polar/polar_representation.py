import numpy as np
from typing import Tuple

from src.image_process.diffraction_pattern import eDiffractionPattern
from src.image_process.polar.polar_transformation import CVPolarTransformation, PolarTransformation
from src.image_process.mask.angular_mask import IAngularMask, MeanAngularMask

                

class PolarRepresentation:
    
    def __init__(self, 
                 edp: eDiffractionPattern,

                 radial_range: Tuple[float, float] = (0, 1),
                 angular_range: Tuple[float, float] = (0, 359),
                 
                 polar_transformer: PolarTransformation = CVPolarTransformation):
        
        self._check_radial_range(radial_range)

        # Initialize parameters and objects
        self._edp = edp

        self._polar_transformer = polar_transformer()

        self._relative_radial_start = radial_range[0]
        self._relative_radial_end = radial_range[1]

        self._start_angle = angular_range[0]
        self._end_angle = angular_range[1]

        self._full_radius_space = None
        self._full_theta_space = None
        self._start_radial_index = None
        self._end_radial_index = None
        self._start_angle_index = None
        self._end_angle_index = None

        self._full_polar_image = None
        self._polar_image = None
        self._radius_space = None
        self._theta_space = None
        self._angular_mask = None

    def _check_radial_range(self, radial_range):
        """
        Checks if the given radial range is valid.

        Raises a ValueError if any of the conditions are not met:

        - relative_radial_start must be greater than zero.
        - relative_radial_end must be less than one.
        - relative_radial_end must be greater than relative_radial_start.

        Parameters:
        radial_range (Tuple[float, float]): Radial range to check.
        """
        start, end = radial_range

        if start<0:
            raise ValueError('relative_radial_start must be greater than zero.')
        if end>1:
            raise ValueError('relative_radial_end must be less than one.')
        if end < start:
            raise ValueError('relative_radial_end must greater than relative_radial_start.')

    # ======== Full data computation
    def _compute_full_polar_image(self):
        if self._full_polar_image is None:  
            self._full_polar_image = self._polar_transformer.transform(self._edp.data, self._edp.center)

    def _compute_full_radius_space(self):
        self._compute_full_polar_image()
        if self._full_radius_space is None:
            self._full_radius_space = np.arange(0, self._full_polar_image.shape[1], 1)

    def _compute_full_theta_space(self):
        self._compute_full_polar_image()
        DEFAULT_MAX_ANGLE = 360
        if self._full_theta_space is None:
            self._full_theta_space = np.linspace(0, DEFAULT_MAX_ANGLE-1, self._full_polar_image.shape[0], endpoint=False)



    # ======== Radial and angular index computation
    def _compute_radial_index(self):
        self._compute_full_radius_space()
        self._start_radial_index = round(self._relative_radial_start * self._full_radius_space.shape[0])
        self._end_radial_index = round(self._relative_radial_end * self._full_radius_space.shape[0])
    
    def _compute_angular_index(self):
        self._compute_full_theta_space()
        DEFAULT_MAX_ANGLE = 360
        self._start_angle_index = int(np.argmin(np.abs(self._full_theta_space - (self._start_angle % DEFAULT_MAX_ANGLE))))
        self._end_angle_index = int(np.argmin(np.abs(self._full_theta_space - (self._end_angle % DEFAULT_MAX_ANGLE))))

    @property
    def radial_range(self):
        return (self._relative_radial_start, self._relative_radial_end)
    
    @property
    def angular_range(self):
        return (self._start_angle, self._end_angle)

    @radial_range.setter
    def radial_range(self, range: Tuple[float, float]) -> None:
        start, end = range

        self._check_radial_range(range)
        
        if start != self._relative_radial_start or end != self._relative_radial_end:
            self._relative_radial_start = start
            self._relative_radial_end = end

            self._compute_radial_index()

    @angular_range.setter
    def angular_range(self, range: Tuple[float, float]) -> None:
        start_angle, end_angle = range

        if start_angle != self._start_angle or end_angle != self._end_angle:
            self._start_angle = start_angle
            self._end_angle = end_angle

            self._compute_angular_index()



    # ======== Polar image and space computation
    def _compute_radius_space(self):
        self._compute_radial_index()
        self._radius_space = self._full_radius_space[self._start_radial_index:self._end_radial_index]

    def _compute_theta_space(self):
        self._compute_angular_index()
        if self._start_angle_index <= self._end_angle_index:
            self._theta_space = self._full_theta_space[self._start_angle_index:self._end_angle_index]
        else:
            self._theta_space = np.concatenate([
                self._full_theta_space[self._start_angle_index:],
                self._full_theta_space[:self._end_angle_index]
            ])

    def _compute_polar_image(self):
        self._compute_radial_index()
        self._compute_angular_index()
        radial_cropped_polar_image = self._full_polar_image[:,self._start_radial_index:self._end_radial_index]

        if self._start_angle_index <= self._end_angle_index:
            self._polar_image = radial_cropped_polar_image[self._start_angle_index:self._end_angle_index, :]
        else:
            self._polar_image = np.concatenate([
                radial_cropped_polar_image[self._start_angle_index:, :],
                radial_cropped_polar_image[:self._end_angle_index, :]
            ])



    # ======== Angular mask computation
    def _compute_angular_mask(self):
        self._compute_radial_index()
        self._compute_angular_index()
        self._compute_polar_image()
        full_angular_mask = self._polar_transformer.transform(self._edp.mask, self._edp.center)[:,self._start_radial_index]

        if self._start_angle_index <= self._end_angle_index:
            self._angular_mask = full_angular_mask[self._start_angle_index:self._end_angle_index]
        else:
            self._angular_mask = np.concatenate([
                full_angular_mask[self._start_angle_index:],
                full_angular_mask[:self._end_angle_index]
            ])

    @property
    def polar_image(self):
        self._compute_polar_image()
        return self._polar_image

    @property
    def radius(self):
        self._compute_radius_space()
        return self._radius_space
    
    @property
    def theta(self):
        self._compute_theta_space()
        return self._theta_space

    @property
    def angular_mask(self):
        self._compute_angular_mask()
        return self._angular_mask


        