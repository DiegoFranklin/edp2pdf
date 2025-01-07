import numpy as np
from typing import Tuple

from src.image_process.diffraction_pattern import eDiffractionPattern
from src.image_process.polar.polar_transformation import CVPolarTransformation, PolarTransformation


class PolarRepresentation:
    """
    Represents a diffraction pattern in polar coordinates.

    This class provides methods to compute and retrieve the polar image, radius space, and theta space of the polar representation.
    It also allows for setting and getting the radial and angular ranges of the polar representation.

    Attributes:
        radial_range (Tuple[float, float]): The radial range of the polar representation.
        angular_range (Tuple[float, float]): The angular range of the polar representation.
        polar_image (np.ndarray): The polar image of the pattern.
        radius (np.ndarray): The radius space of the polar representation.
        theta (np.ndarray): The theta space of the polar representation.
        angular_mask (np.ndarray): The angular mask of the polar representation.
        polar_mask (np.ndarray): The polar mask of the polar representation.
    """

    def __init__(
        self,
        edp: eDiffractionPattern,
        radial_range: Tuple[float, float] = (0, 1),
        angular_range: Tuple[float, float] = (0, 359),
        polar_transformer: PolarTransformation = CVPolarTransformation,
    ):
        """
        Initializes the PolarRepresentation object.

        Args:
            edp (eDiffractionPattern): The diffraction pattern to represent in polar coordinates.
            radial_range (Tuple[float, float], optional): The radial range of the polar representation. Defaults to (0, 1).
            angular_range (Tuple[float, float], optional): The angular range of the polar representation. Defaults to (0, 359).
            polar_transformer (PolarTransformation, optional): The polar transformer to convert from cartesian to polar coordinates.
                                                              Defaults to CVPolarTransformation.

        Raises:
            TypeError: If `edp` is not an instance of eDiffractionPattern or if `polar_transformer` is not a subclass of PolarTransformation.
            ValueError: If `radial_range` or `angular_range` are invalid.
        """
        if not isinstance(edp, eDiffractionPattern):
            raise TypeError("Input `edp` must be an instance of eDiffractionPattern.")
        if not issubclass(polar_transformer, PolarTransformation):
            raise TypeError("Input `polar_transformer` must be a subclass of PolarTransformation.")

        self._check_radial_range(radial_range)
        self._check_angular_range(angular_range)

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
        self._extended_polar_mask = None
        self._polar_mask = None
        self._angular_mask = None

    def _check_radial_range(self, radial_range: Tuple[float, float]):
        """
        Checks if the given radial range is valid.

        Args:
            radial_range (Tuple[float, float]): Radial range to check.

        Raises:
            ValueError: If `radial_range` is invalid.
        """
        start, end = radial_range

        if start < 0:
            raise ValueError("relative_radial_start must be greater than zero.")
        if end > 1:
            raise ValueError("relative_radial_end must be less than one.")
        if end < start:
            raise ValueError("relative_radial_end must be greater than relative_radial_start.")

    def _check_angular_range(self, angular_range: Tuple[float, float]):
        """
        Checks if the given angular range is valid.

        Args:
            angular_range (Tuple[float, float]): Angular range to check.

        Raises:
            ValueError: If `angular_range` is invalid.
        """
        start, end = angular_range

        if start < 0 or end < 0:
            raise ValueError("Angular range values must be non-negative.")
        if end < start:
            raise ValueError("Angular range end must be greater than or equal to start.")

    def _compute_full_polar_image(self):
        """Computes the full polar image if not already computed."""
        if self._full_polar_image is None:
            self._full_polar_image = self._polar_transformer.transform(self.edp.data, self.edp.center)

    def _compute_full_radius_space(self):
        """Computes the full radius space if not already computed."""
        self._compute_full_polar_image()
        if self._full_radius_space is None:
            self._full_radius_space = np.arange(0, self._full_polar_image.shape[1], 1)

    def _compute_full_theta_space(self):
        """Computes the full theta space if not already computed."""
        self._compute_full_polar_image()
        DEFAULT_MAX_ANGLE = 360
        if self._full_theta_space is None:
            self._full_theta_space = np.linspace(0, DEFAULT_MAX_ANGLE - 1, self._full_polar_image.shape[0], endpoint=False)

    def _compute_extended_polar_mask(self):
        """Computes the extended polar mask for rotational integration."""
        if self._extended_polar_mask is None:
            if self.edp.mask is None:
                raise ValueError("EDP does not have a mask.")
            try:
                mask_edp = eDiffractionPattern(
                    data=self.edp.mask,
                    center=self.edp.center,
                    mask=self.edp.mask,
                )

                mask_polar = PolarRepresentation(mask_edp)
                mask_polar.radial_range = self._relative_radial_range

                full_edp = eDiffractionPattern(
                    data=np.ones_like(self.edp.mask),
                    center=self.edp.center,
                    mask=np.ones_like(self.edp.mask),
                )

                full_polar = PolarRepresentation(full_edp)
                full_polar.radial_range = self._relative_radial_range

                polar_mask = mask_polar.polar_image.copy()
                lock = np.argmax(np.sum(polar_mask, axis=0))
                lock = round(lock * 0.9)

                last_angular_mask = polar_mask[:, lock - 1]
                for i in range(0, round(lock * 0.05)):
                    last_angular_mask = np.logical_or(last_angular_mask, polar_mask[:, lock - 1 - i])

                full_angular_mask = np.tile(last_angular_mask, (full_polar.polar_image.shape[1], 1)).T

                polar_mask[:, lock:] = 0

                polar_full = full_polar.polar_image.copy()
                polar_full = np.logical_and(polar_full, full_angular_mask)
                polar_full[:, :lock] = 0

                final_mask = np.logical_or(polar_mask, polar_full)

                self._extended_polar_mask = final_mask
            except Exception as e:
                raise RuntimeError(f"Failed to construct extended polar mask: {e}")

    def _compute_radial_index(self):
        """
        Computes the radial indices for the specified radial range.

        Modifies:
            self._radial_indices (Tuple[int, int]): Indices corresponding to the start and end of the specified radial range.
        """
        self._compute_full_radius_space()
        start, end = self._relative_radial_range
        total_radii = self._full_radius_space.shape[0]
        self._radial_indices = (round(start * total_radii), round(end * total_radii))

    def _compute_angular_index(self):
        """
        Computes the angular indices for the specified angular range.

        Modifies:
            self._angular_indices (Tuple[int, int]): Indices corresponding to the start and end of the specified angular range.
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
        """Gets the radial range of the polar representation."""
        return self._relative_radial_range

    @radial_range.setter
    def radial_range(self, range: Tuple[float, float]) -> None:
        """
        Sets the radial range of the polar representation.

        Args:
            range (Tuple[float, float]): The new radial range.

        Raises:
            ValueError: If the new radial range is invalid.
        """
        start, end = range
        self._check_radial_range(range)

        if (start, end) != self._relative_radial_range:
            self._relative_radial_range = (start, end)
            self._compute_radial_index()

    @property
    def angular_range(self) -> Tuple[float, float]:
        """Gets the angular range of the polar representation."""
        return self._angular_range

    @angular_range.setter
    def angular_range(self, range: Tuple[float, float]) -> None:
        """
        Sets the angular range of the polar representation.

        Args:
            range (Tuple[float, float]): The new angular range.

        Raises:
            ValueError: If the new angular range is invalid.
        """
        start_angle, end_angle = range
        self._check_angular_range(range)

        if (start_angle, end_angle) != self._angular_range:
            self._angular_range = (start_angle, end_angle)
            self._compute_angular_index()

    def _compute_radius_space(self):
        """Computes the radius space for the current radial range."""
        self._compute_radial_index()
        start_idx, end_idx = self._radial_indices
        self._radius_space = self._full_radius_space[start_idx:end_idx]

    def _compute_theta_space(self):
        """Computes the theta space for the current angular range."""
        self._compute_angular_index()
        start_idx, end_idx = self._angular_indices
        if start_idx <= end_idx:
            self._theta_space = self._full_theta_space[start_idx:end_idx]
        else:
            self._theta_space = np.concatenate([self._full_theta_space[start_idx:], self._full_theta_space[:end_idx]])

    def _compute_polar_image(self):
        """Computes the polar image for the current radial and angular ranges."""
        self._compute_radial_index()
        self._compute_angular_index()
        start_r, end_r = self._radial_indices
        start_a, end_a = self._angular_indices

        radial_cropped_polar_image = self._full_polar_image[:, start_r:end_r]

        if start_a <= end_a:
            self._polar_image = radial_cropped_polar_image[start_a:end_a, :]
        else:
            self._polar_image = np.concatenate([radial_cropped_polar_image[start_a:, :], radial_cropped_polar_image[:end_a, :]])

    def _compute_polar_mask(self):
        """Computes the polar mask for the current polar representation."""
        self._compute_extended_polar_mask()
        self._compute_radial_index()
        self._compute_angular_index()
        start_r, end_r = self._radial_indices
        start_a, end_a = self._angular_indices

        full_polar_mask = self._extended_polar_mask[:, start_r:end_r]

        if start_a <= end_a:
            self._polar_mask = full_polar_mask[start_a:end_a, :]
        else:
            self._polar_mask = np.concatenate([full_polar_mask[start_a:, :], full_polar_mask[:end_a, :]])

    def _compute_angular_mask(self):
        """Computes the angular mask for the current polar representation."""
        self._compute_extended_polar_mask()
        self._compute_radial_index()
        self._compute_angular_index()
        start_r, _ = self._radial_indices
        start_a, end_a = self._angular_indices

        full_angular_mask = self._extended_polar_mask[:, start_r]

        if start_a <= end_a:
            self._angular_mask = full_angular_mask[start_a:end_a]
        else:
            self._angular_mask = np.concatenate([full_angular_mask[start_a:], full_angular_mask[:end_a]])

    @property
    def polar_image(self) -> np.ndarray:
        """Gets the polar image of the pattern."""
        self._compute_polar_image()
        return self._polar_image

    @property
    def radius(self) -> np.ndarray:
        """Gets the radius space of the polar representation."""
        self._compute_radius_space()
        return self._radius_space

    @property
    def theta(self) -> np.ndarray:
        """Gets the theta space of the polar representation."""
        self._compute_theta_space()
        return self._theta_space

    @property
    def polar_mask(self) -> np.ndarray:
        """
        Gets the polar mask of the polar representation.

        The polar mask is derived from the extended polar mask and is cropped to the current radial and angular ranges.

        Returns:
            np.ndarray: The polar mask as a binary numpy array.
        """
        self._compute_polar_mask()
        return self._polar_mask

    @property
    def angular_mask(self) -> np.ndarray:
        """Gets the angular mask of the polar representation."""
        self._compute_angular_mask()
        return self._angular_mask