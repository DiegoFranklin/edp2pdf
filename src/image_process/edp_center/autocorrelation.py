import numpy as np
import cv2
from typing import Tuple

class AutoCorrelation:
    def __init__(self, data: np.ndarray, mask: np.ndarray):
        self.data = data
        self.mask = np.asarray(mask).astype("float")

    def compute(self) -> Tuple[Tuple[int, int], np.ndarray]:
        masked_image = self.data * self.mask
        paded_masked_image = np.pad(masked_image, ((0, self.data.shape[0]), (0, self.data.shape[1])))
        paded_mask = np.pad(self.mask, ((0, self.mask.shape[0]), (0, self.mask.shape[1])), constant_values=(1.0, 1.0))
        
        pm_fft = np.fft.fft2(paded_mask)
        part1 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image) ** 2) * np.fft.ifft2(pm_fft ** 2))
        part2 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image ** 2) * pm_fft))
        part3 = np.real(np.fft.ifft2(np.fft.fft2(paded_masked_image * paded_mask)))
        
        autocorrelation =  (part1 - part3) / (part2 - part3)

        cj = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)[0]/2 % self.data.shape[0]
        ci = np.unravel_index(np.argmax(autocorrelation), autocorrelation.shape)[1]/2 % self.data.shape[1]



        autocorrelation = autocorrelation - np.min(autocorrelation)
        autocorrelation = autocorrelation/np.sum(autocorrelation)

        self.resized_autocorrelation = cv2.resize(autocorrelation, self.data.shape)
        self.ac_center = np.unravel_index(np.argmax(self.resized_autocorrelation), self.resized_autocorrelation.shape)

        return (round(ci), round(cj)), self.resized_autocorrelation
