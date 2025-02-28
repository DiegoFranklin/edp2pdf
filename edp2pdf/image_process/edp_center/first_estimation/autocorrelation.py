import numpy as np
import cv2
from typing import Tuple
from edp2pdf.image_process.edp_center.first_estimation.validate_inputs import validate_inputs

def apply_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Applies the mask to the image data."""
    return data * mask

def pad_image(image: np.ndarray,
              target_shape: Tuple[int, int],
              constant_values: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Pads the image to the target shape."""
    return np.pad(image, ((0, target_shape[0]), (0, target_shape[1])), constant_values=constant_values)

def compute_autocorrelation(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Computes the autocorrelation of the masked image."""
    masked_image = apply_mask(data, mask)
    paded_masked_image = pad_image(masked_image, masked_image.shape)
    paded_mask = pad_image(mask, mask.shape, constant_values=(1.0, 1.0))

    pm_fft = np.fft.fft2(paded_mask)
    part1 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image) ** 2) * np.fft.ifft2(pm_fft ** 2))
    part2 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image ** 2) * pm_fft))
    part3 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image * paded_mask)))

    autocorrelation = (part1 - part3) / (part2 - part3)
    return autocorrelation

def find_center(autocorrelation: np.ndarray, data_shape: Tuple[int, int]) -> Tuple[int, int]:
    """Finds the center coordinates of the autocorrelation."""
    cj = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)[0] / 2 % data_shape[0]
    ci = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)[1] / 2 % data_shape[1]
    return round(ci), round(cj)

def autocorrelation(data: np.ndarray, mask: np.ndarray) -> Tuple[Tuple[int, int], np.ndarray]:
    """Computes the autocorrelation of the masked image."""
    validate_inputs(data=data, mask=mask)
    autocorrelation = compute_autocorrelation(data, mask)
    ci, cj = find_center(autocorrelation, data.shape)
    return (ci, cj)
