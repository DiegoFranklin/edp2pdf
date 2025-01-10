import numpy as np
import cv2
from typing import Tuple


class AutoCorrelation:
    """
    A class for computing the autocorrelation of an image with a given mask.

    Attributes:
        data (np.ndarray): The input image data.
        mask (np.ndarray): The binary mask applied to the image.
        resized_autocorrelation (np.ndarray): The resized autocorrelation result.
        ac_center (Tuple[int, int]): The center coordinates of the autocorrelation.
    """

    def __init__(self, data: np.ndarray, mask: np.ndarray):
        """
        Initializes the AutoCorrelation class.

        Args:
            data (np.ndarray): The input image data.
            mask (np.ndarray): The binary mask applied to the image.

        Raises:
            ValueError: If `data` and `mask` have incompatible shapes.
            TypeError: If `data` or `mask` are not numpy arrays.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if not isinstance(mask, np.ndarray):
            raise TypeError("Input `mask` must be a numpy array.")
        if data.shape != mask.shape:
            raise ValueError("Shapes of `data` and `mask` must match.")

        self.data = data
        self.mask = np.asarray(mask).astype("float")

    def compute(self) -> Tuple[Tuple[int, int], np.ndarray]:
        """
        Computes the autocorrelation of the masked image.

        Returns:
            Tuple[Tuple[int, int], np.ndarray]: A tuple containing:
                - The center coordinates of the autocorrelation (ci, cj).
                - The resized autocorrelation result.

        Raises:
            RuntimeError: If the autocorrelation computation fails.
        """
        try:
            # Apply mask to the image
            masked_image = self.data * self.mask

            # Pad the masked image and mask
            paded_masked_image = np.pad(masked_image, ((0, self.data.shape[0]), (0, self.data.shape[1])))
            paded_mask = np.pad(self.mask, ((0, self.mask.shape[0]), (0, self.mask.shape[1])), constant_values=(1.0, 1.0))

            # Compute FFTs and intermediate results
            pm_fft = np.fft.fft2(paded_mask)
            part1 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image) ** 2) * np.fft.ifft2(pm_fft ** 2))
            part2 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image ** 2) * pm_fft))
            part3 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image * paded_mask)))

            # Compute autocorrelation
            autocorrelation = (part1 - part3) / (part2 - part3)

            # Find the center coordinates
            cj = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)[0] / 2 % self.data.shape[0]
            ci = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)[1] / 2 % self.data.shape[1]

            # Normalize the autocorrelation
            autocorrelation = autocorrelation - np.min(autocorrelation)
            autocorrelation = autocorrelation / np.sum(autocorrelation)

            # Resize the autocorrelation to match the original data shape
            self.resized_autocorrelation = cv2.resize(autocorrelation, self.data.shape)
            self.ac_center = np.unravel_index(np.argmax(self.resized_autocorrelation), self.resized_autocorrelation.shape)

            return (round(ci), round(cj)), self.resized_autocorrelation

        except Exception as e:
            raise RuntimeError(f"Autocorrelation computation failed: {e}")