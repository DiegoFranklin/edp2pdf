from ..diffraction_pattern import eDiffractionPattern

from typing import Tuple
import numpy as np
import cv2

class EllipseCorrector:
    def __init__(self, correction_params: Tuple):
        self.orientation = correction_params[0]
        self.sc_fact = correction_params[1]


    def correct(self, edp: eDiffractionPattern):
        data = edp.data

        l = round(np.sqrt(data.shape[0]**2 + data.shape[1]**2) / 2)

        new_center = (np.asarray((l, l)) + np.asarray(edp.center)).astype(np.float32)

        data_for_rot = np.pad(data.copy(), ((l, l), (l, l)), mode='constant', constant_values=0)

        M_semi = cv2.getRotationMatrix2D(tuple(new_center), self.orientation, 1.0)
        semi_rotated = cv2.warpAffine(data_for_rot.copy(), M_semi, data_for_rot.shape)

        semi_corrected = cv2.resize(semi_rotated, (round(semi_rotated.shape[0]*self.sc_fact), semi_rotated.shape[1]), interpolation=cv2.INTER_CUBIC)

        M = cv2.getRotationMatrix2D(tuple(new_center), 90, 1.0)
        rotated = cv2.warpAffine(semi_corrected, M, data_for_rot.shape)

        corrected = cv2.resize(rotated, (round(rotated.shape[0]/self.sc_fact), rotated.shape[1]), interpolation=cv2.INTER_CUBIC)

        M_final = cv2.getRotationMatrix2D(tuple(new_center), -(self.orientation+90), 1.0)
        final_al = cv2.warpAffine(corrected.copy(), M_final, data_for_rot.shape)

        final = final_al[l:l+data.shape[0], l:l+data.shape[0]]

        corrected_edp = eDiffractionPattern(data=final, center=None)

        return corrected_edp