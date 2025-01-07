import numpy as np

from .polar_representation import PolarRepresentation
from src.image_process.diffraction_pattern import eDiffractionPattern


class RotationalIntegration:
    """
    A class for performing rotational integration on a polar representation of a diffraction pattern.

    This class provides methods to compute rotational integration over a specified angular range,
    either with or without masking.

    Attributes:
        _polar_representation (PolarRepresentation): The polar representation of the diffraction pattern.
        extended_polar_mask (np.ndarray): The extended polar mask used for masked integration.
    """

    def __init__(self, polar_representation: PolarRepresentation):
        """
        Initializes the RotationalIntegration object.

        Args:
            polar_representation (PolarRepresentation): The polar representation of the diffraction pattern.

        Raises:
            TypeError: If `polar_representation` is not an instance of PolarRepresentation.
        """
        if not isinstance(polar_representation, PolarRepresentation):
            raise TypeError("Input `polar_representation` must be an instance of PolarRepresentation.")

        self._polar_representation = polar_representation
        self.extended_polar_mask = None

    def _construct_extended_polar_mask(self) -> np.ndarray:
        """
        Constructs an extended polar mask for rotational integration.

        The mask is constructed by combining the original polar mask with a synthetic mask
        to handle edge cases and ensure smooth integration.

        Returns:
            np.ndarray: The extended polar mask as a binary numpy array.

        Raises:
            RuntimeError: If the mask construction fails.
        """
        try:
            mask_edp = eDiffractionPattern(
                data=self._polar_representation.edp.mask,
                center=self._polar_representation.edp.center,
                mask=self._polar_representation.edp.mask,
            )

            mask_polar = PolarRepresentation(mask_edp)
            mask_polar.radial_range = self._polar_representation.radial_range

            full_edp = eDiffractionPattern(
                data=np.ones_like(self._polar_representation.edp.mask),
                center=self._polar_representation.edp.center,
                mask=np.ones_like(self._polar_representation.edp.mask),
            )

            full_polar = PolarRepresentation(full_edp)
            full_polar.radial_range = self._polar_representation.radial_range

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

            return final_mask
        except Exception as e:
            raise RuntimeError(f"Failed to construct extended polar mask: {e}")

    def get_masked_rotational_integration(
        self, start_angle: float, end_angle: float, method: str = "mean"
    ) -> np.ndarray:
        """
        Computes the rotational integration over a specified angular range using a mask.

        Args:
            start_angle (float): The starting angle for the integration range (in degrees).
            end_angle (float): The ending angle for the integration range (in degrees).
            method (str, optional): The integration method. Can be "mean" or "median". Defaults to "mean".

        Returns:
            np.ndarray: The integrated values as a numpy array.

        Raises:
            ValueError: If the method is invalid or if the angular range is invalid.
            RuntimeError: If the integration fails.
        """
        if method not in {"mean", "median"}:
            raise ValueError(f"Invalid method: {method}. Supported methods are 'mean' and 'median'.")

        try:
            if self.extended_polar_mask is None:
                self.extended_polar_mask = self._construct_extended_polar_mask()

            self._polar_representation.angular_range = (start_angle, end_angle)

            polar_data = self._polar_representation.polar_image
            final_mask = self.extended_polar_mask
            integral = []
            for i in range(polar_data.shape[1]):
                col = polar_data[:, i]
                col_mask = final_mask[:, i]

                if method == "median":
                    integral.append(np.median(col[col_mask == 1]))
                elif method == "mean":
                    integral.append(np.mean(col[col_mask == 1]))

            return np.array(integral)
        except Exception as e:
            raise RuntimeError(f"Failed to compute masked rotational integration: {e}")

    def get_rotational_integration(
        self, start_angle: float, end_angle: float, method: str = "mean"
    ) -> np.ndarray:
        """
        Computes the rotational integration over a specified angular range without masking.

        Args:
            start_angle (float): The starting angle for the integration range (in degrees).
            end_angle (float): The ending angle for the integration range (in degrees).
            method (str, optional): The integration method. Can be "mean" or "median". Defaults to "mean".

        Returns:
            np.ndarray: The integrated values as a numpy array.

        Raises:
            ValueError: If the method is invalid or if the angular range is invalid.
            RuntimeError: If the integration fails.
        """
        if method not in {"mean", "median"}:
            raise ValueError(f"Invalid method: {method}. Supported methods are 'mean' and 'median'.")

        try:
            self._polar_representation.angular_range = (start_angle, end_angle)
            if method == "mean":
                return np.mean(self._polar_representation.polar_image, axis=0)
            if method == "median":
                return np.median(self._polar_representation.polar_image, axis=0)
        except Exception as e:
            raise RuntimeError(f"Failed to compute rotational integration: {e}")