import numpy as np

from .polar_representation import PolarRepresentation
from edp2pdf.image_process.diffraction_pattern import eDiffractionPattern


class RotationalIntegration:
    """
    A class for performing rotational integration on a polar representation of a diffraction pattern.

    This class provides a method to compute rotational integration over a specified angular range,
    with an option to use a mask.

    Attributes:
        _polar_representation (PolarRepresentation): The polar representation of the diffraction pattern.
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

    def get_rotational_integration(
        self, start_angle: float, end_angle: float, method: str = "mean", use_mask: bool = False
    ) -> np.ndarray:
        """
        Computes the rotational integration over a specified angular range.

        Args:
            start_angle (float): The starting angle for the integration range (in degrees).
            end_angle (float): The ending angle for the integration range (in degrees).
            method (str, optional): The integration method. Can be "mean" or "median". Defaults to "mean".
            use_mask (bool, optional): Whether to use the polar mask for integration. Defaults to False.

        Returns:
            np.ndarray: The integrated values as a numpy array.

        Raises:
            ValueError: If the method is invalid or if the angular range is invalid.
            RuntimeError: If the integration fails.
        """
        if method not in {"mean", "median"}:
            raise ValueError(f"Invalid method: {method}. Supported methods are 'mean' and 'median'.")

        try:
            # Set the angular range for the polar representation
            self._polar_representation.angular_range = (start_angle, end_angle)

            # Get the polar image
            polar_data = self._polar_representation.polar_image

            if use_mask:
                # Get the polar mask for masked integration
                polar_mask = self._polar_representation.polar_mask

                # Perform masked integration along the angular axis
                integral = []
                for i in range(polar_data.shape[1]):  # Iterate over radial bins
                    col = polar_data[:, i]  # Extract the column (angular data for this radial bin)
                    col_mask = polar_mask[:, i]  # Extract the corresponding mask

                    if method == "median":
                        integral.append(np.median(col[col_mask == 1]))  # Apply mask and compute median
                    elif method == "mean":
                        integral.append(np.mean(col[col_mask == 1]))  # Apply mask and compute mean

                return np.array(integral)
            else:
                # Perform unmasked integration along the angular axis
                if method == "mean":
                    return np.mean(polar_data, axis=0)  # Mean along the angular axis
                elif method == "median":
                    return np.median(polar_data, axis=0)  # Median along the angular axis
        except Exception as e:
            raise RuntimeError(f"Failed to compute rotational integration: {e}")