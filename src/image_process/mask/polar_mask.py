import numpy as np

from image_process.polar.polar_representation import PolarRepresentation
from image_process.mask.mask_getters import MeanTreshMask, MaskGetter
from image_process.mask.mask_ops import Dilate, Erode, OCRoutine

class PolarLineMask:
    def __init__(self, polar_representation: PolarRepresentation):
        self._polar_representation = polar_representation

    def get_polar_line_mask(
            self,
            angular_range_expansion: float = None,
            cyclic_shift: int = None,
            radius_num: int = 20,
            mask_getter: MaskGetter = None,
            ocroutine: OCRoutine = None) -> np.array:
        """
        Retrieves a polar line mask from the polar representation data.

        Parameters:
        - radius_num: The index of the radius to extract from polar data.
        - cyclic_shift: Optional cyclic shift in degrees for aligning the mask.
        - mask_getter: Optional custom mask getter instance.
        - ocroutine: Optional custom OCRoutine instance for morphological operations.

        Returns:
        - A numpy array representing the polar line mask after applying the mask and operations.
        """
        # Ensure the polar representation is valid
        if not hasattr(self._polar_representation, 'polar_data') or not hasattr(self._polar_representation, 'theta'):
            raise ValueError("Invalid PolarRepresentation object.")

        # Extract the polar data
        self._polar_data = self._polar_representation.polar_data
        if radius_num >= self._polar_data.shape[1]:
            raise IndexError("radius_num exceeds the number of available radii in polar data.")

        # Extract the data section at the specified radius
        data_section = self._polar_data[:, :radius_num]

        # Apply the mask getter
        mask_getter = mask_getter or MeanTreshMask(constant=0)
        mask_section = mask_getter.get_mask(data_section)

        # Define and apply the OCRoutine
        ocroutine = ocroutine or OCRoutine([Erode(iterations=5, kernel_size=2), Dilate(iterations=5, kernel_size=2)])
        mask_section = ocroutine.execute_routine(mask_section)

        # Extract the first column of the mask after operations
        polar_line_mask = mask_section[:, 0]


        if angular_range_expansion is not None:
            num_beams = round(angular_range_expansion*self._polar_representation.theta.shape[0]/360)

            expanded_hole_mask = polar_line_mask.copy()
            for i in range(-num_beams, num_beams):
                rolled_mask = np.roll(polar_line_mask, shift=i)
                expanded_hole_mask = np.logical_and(expanded_hole_mask, rolled_mask)

            polar_line_mask = expanded_hole_mask.copy()


        # Apply cyclic shift if needed
        if cyclic_shift is not None:
            shift = np.argmin(np.abs((cyclic_shift % 360) - self._polar_representation.theta))
            polar_line_mask = np.logical_and(polar_line_mask, np.roll(polar_line_mask, shift))

        return polar_line_mask
