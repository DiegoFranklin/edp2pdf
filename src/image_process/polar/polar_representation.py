import numpy as np
from typing import Tuple

from src.image_process.diffraction_pattern import eDiffractionPattern
from src.image_process.polar.polar_transformation import CVPolarTransformation, PolarTransformation

                

class PolarRepresentation:
    """
    Represents a diffraction pattern in polar coordinates.

    This class provides methods to compute and retrieve the polar image, radius space, and theta space of the polar representation.
    It also allows for setting and getting the radial and angular ranges of the polar representation.

    Parameters:
    edp (eDiffractionPattern): The diffraction pattern to represent in polar coordinates.
    radial_range (Tuple[float, float], optional): The radial range of the polar representation. Defaults to (0, 1).
    angular_range (Tuple[float, float], optional): The angular range of the polar representation. Defaults to (0, 359).
    polar_transformer (PolarTransformation, optional): The polar transformer to convert from cartesian to polar coordinates. 
    Defaults to CVPolarTransformation.

    Attributes:
    radial_range (Tuple[float, float]): The radial range of the polar representation.
    angular_range (Tuple[float, float]): The angular range of the polar representation.
    polar_image (numpy.ndarray): The polar image of the pattern.
    radius (numpy.ndarray): The radius space of the polar representation.
    theta (numpy.ndarray): The theta space of the polar representation.
    angular_mask (numpy.ndarray): The angular mask of the polar representation.

    Methods:
    radial_range: Gets or sets the radial range of the polar representation.
    angular_range: Gets or sets the angular range of the polar representation.
    polar_image: Gets the polar image of the pattern.
    radius: Gets the radius space of the polar representation.
    theta: Gets the theta space of the polar representation.
    angular_mask: Gets the angular mask of the polar representation.
    """
    
    def __init__(self, 
                 edp: eDiffractionPattern,

                 radial_range: Tuple[float, float] = (0, 1),
                 angular_range: Tuple[float, float] = (0, 359),
                 
                 polar_transformer: PolarTransformation = CVPolarTransformation):
        
        self._check_radial_range(radial_range)

        self.edp = edp

        self._polar_transformer = polar_transformer()

        self._relative_radial_range = radial_range
        self._angular_range = angular_range 

        self._radial_indices = (None, None)
        self._angular_indices = (None, None)

        self._full_polar_image = None
        self._full_radius_space = None
        self._full_theta_space = None

        self._polar_image = None
        self._radius_space = None
        self._theta_space = None
        self._angular_mask = None

    def _check_radial_range(self, radial_range: Tuple[float, float]):
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

        if start < 0:
            raise ValueError('relative_radial_start must be greater than zero.')
        if end > 1:
            raise ValueError('relative_radial_end must be less than one.')
        if end < start:
            raise ValueError('relative_radial_end must be greater than relative_radial_start.')

    # ======== Full data computation
    def _compute_full_polar_image(self):
        if self._full_polar_image is None:  
            self._full_polar_image = self._polar_transformer.transform(self.edp.data, self.edp.center)

    def _compute_full_radius_space(self):
        self._compute_full_polar_image()
        if self._full_radius_space is None:
            self._full_radius_space = np.arange(0, self._full_polar_image.shape[1], 1)

    def _compute_full_theta_space(self):
        self._compute_full_polar_image()
        DEFAULT_MAX_ANGLE = 360
        if self._full_theta_space is None:
            self._full_theta_space = np.linspace(0, DEFAULT_MAX_ANGLE - 1, self._full_polar_image.shape[0], endpoint=False)

    # ======== Radial and angular index computation
    def _compute_radial_index(self):
        """
        Computes the radial indices for the specified radial range.

        This function calculates the start and end indices in the full radius space
        based on the relative radial range, and assigns them to self._radial_indices.

        Parameters:
        self._relative_radial_range (Tuple[float, float]): Radial range specified as a 
        fraction of the entire radius space.

        Modifies:
        self._radial_indices (Tuple[int, int]): Indices corresponding to the start and 
        end of the specified radial range in the full radius space.
        """
        self._compute_full_radius_space()
        start, end = self._relative_radial_range
        total_radii = self._full_radius_space.shape[0]
        self._radial_indices = (
            round(start * total_radii),
            round(end * total_radii)
        )
    
    def _compute_angular_index(self):
        """
        Computes the angular indices for the specified angular range.

        This function calculates the start and end indices in the full theta space
        based on the specified angular range, and assigns them to self._angular_indices.

        Parameters:
        self._angular_range (Tuple[float, float]): Angular range specified in degrees.

        Modifies:
        self._angular_indices (Tuple[int, int]): Indices corresponding to the start and end of the specified angular range in the full theta space.
        """
        self._compute_full_theta_space()
        DEFAULT_MAX_ANGLE = 360
        start_angle, end_angle = self._angular_range
        full_theta = self._full_theta_space

        start_idx = int(np.argmin(np.abs(full_theta - (start_angle % DEFAULT_MAX_ANGLE))))
        end_idx = int(np.argmin(np.abs(full_theta - (end_angle % DEFAULT_MAX_ANGLE))))
        self._angular_indices = (start_idx, end_idx)

    @property
    def radial_range(self) -> Tuple[float, float]:
        return self._relative_radial_range
    
    @radial_range.setter
    def radial_range(self, range: Tuple[float, float]) -> None:
        start, end = range

        self._check_radial_range(range)
        
        if (start, end) != self._relative_radial_range:
            self._relative_radial_range = (start, end)
            self._compute_radial_index()

    @property
    def angular_range(self) -> Tuple[float, float]:
        return self._angular_range
    
    @angular_range.setter
    def angular_range(self, range: Tuple[float, float]) -> None:
        start_angle, end_angle = range

        if (start_angle, end_angle) != self._angular_range:
            self._angular_range = (start_angle, end_angle)
            self._compute_angular_index()

    # ======== Polar image and space computation
    def _compute_radius_space(self):
        self._compute_radial_index()
        start_idx, end_idx = self._radial_indices
        self._radius_space = self._full_radius_space[start_idx:end_idx]

    def _compute_theta_space(self):
        self._compute_angular_index()
        start_idx, end_idx = self._angular_indices
        if start_idx <= end_idx:
            self._theta_space = self._full_theta_space[start_idx:end_idx]
        else:
            self._theta_space = np.concatenate([
                self._full_theta_space[start_idx:], 
                self._full_theta_space[:end_idx]
            ])

    def _compute_polar_image(self):
        self._compute_radial_index()
        self._compute_angular_index()
        start_r, end_r = self._radial_indices
        start_a, end_a = self._angular_indices

        radial_cropped_polar_image = self._full_polar_image[:, start_r:end_r]

        if start_a <= end_a:
            self._polar_image = radial_cropped_polar_image[start_a:end_a, :]
        else:
            self._polar_image = np.concatenate([
                radial_cropped_polar_image[start_a:, :],
                radial_cropped_polar_image[:end_a, :]
            ])

    # ======== Angular mask computation
    def _compute_angular_mask(self):
        """
        Computes the angular mask for the current polar representation based on the
        specified radial and angular indices.

        This method calculates the angular mask by transforming the eDiffractionPattern's
        mask to a polar representation and cropping it according to the current radial and
        angular indices. The computed angular mask is stored in the instance variable
        self._angular_mask.

        Modifies:
        self._angular_mask: A 1-dimensional numpy array representing the cropped angular mask
        for the specified angular range.

        Preconditions:
        - self._radial_indices must be computed and valid.
        - self._angular_indices must be computed and valid.
        """
        self._compute_radial_index()
        self._compute_angular_index()
        self._compute_polar_image()
        start_r, _ = self._radial_indices
        start_a, end_a = self._angular_indices

        full_angular_mask = self._polar_transformer.transform(self.edp.mask, self.edp.center)[:, start_r]

        if start_a <= end_a:
            self._angular_mask = full_angular_mask[start_a:end_a]
        else:
            self._angular_mask = np.concatenate([
                full_angular_mask[start_a:], 
                full_angular_mask[:end_a]
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


        