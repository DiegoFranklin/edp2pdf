import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import List, Tuple
from src.image_process.utils import ImagePadder
from src.image_process.diffraction_pattern import eDiffractionPattern


class CorrectionOperation(ABC):
    @abstractmethod
    def operate(self, data: np.ndarray) -> np.ndarray:
        """
        Abstract method to perform an operation on the given data.

        :param data: Input data as a numpy array.
        :return: The result of the operation as a numpy array.
        """
        pass

class Rotate(CorrectionOperation):
    def __init__(self, angle: float, pivot_point: Tuple[int, int]):
        self._angle = angle # degrees
        self._pivot_point = tuple(map(int,pivot_point))

    def operate(self, data: np.ndarray) -> np.ndarray:
        """
        Rotate the given data by the specified angle.

        Parameters
        ----------
        data : np.ndarray
            Input data to be rotated.

        Returns
        -------
        np.ndarray
            Rotated data.
        """
        rotation_matrix = cv2.getRotationMatrix2D(self._pivot_point, self._angle, 1.0)
        rotated_data = cv2.warpAffine(data.copy(), rotation_matrix, data.shape)

        return rotated_data


class Scale(CorrectionOperation):
    def __init__(self, scale_factor: float, interpolation_method: int = cv2.INTER_CUBIC):
        self._scale_factor = scale_factor
        self._interpolation_method = interpolation_method

    def operate(self, data: np.ndarray) -> np.ndarray:
        """
        Scales the input data by the specified scale factor.

        Parameters
        ----------
        data : np.ndarray
            Input data to be scaled.

        Returns
        -------
        scaled_data : np.ndarray
            Scaled data.
        """
        scaled_data: np.ndarray = cv2.resize(data, (round(data.shape[0] * self._scale_factor), data.shape[1]),
                                             interpolation=self._interpolation_method)
        
        return scaled_data

class AddBorder(CorrectionOperation):
    def __init__(self, data_shape: tuple):
        self.border_size = None

        self._compute_border_size(data_shape)

    def _compute_border_size(self, data_shape: np.ndarray):
        self.border_size = (np.linalg.norm(data_shape) - np.max(data_shape)) // 2
    
    def operate(self, data: np.ndarray) -> np.ndarray:
        """
        Compute the border size required to make the data square and pads the data with zeros.

        Parameters
        ----------
        data : np.ndarray
            Input data to be padded.

        Returns
        -------
        padded_data : np.ndarray
            Padded data.
        """

        return np.pad(data, self.border_size, mode='constant', constant_values=0)
    
class RemoveBorder(CorrectionOperation):
    def __init__(self, border_size: int):
        self.border_size = border_size

    def operate(self, data: np.ndarray) -> np.ndarray:
        """
        Removes the border added by the AddBorder operation.

        Parameters
        ----------
        data : np.ndarray
            Input data to be cropped.

        Returns
        -------
        np.ndarray
            Cropped data.
        """
        return data[self.border_size:-self.border_size, self.border_size:-self.border_size]

   



class EllipseCorrection:
    def __init__(self, elp_params: dict):
        self._elp_params = elp_params

        self._amplitude = None
        self._orientation = None

        self._manage_elp_params()

    def _manage_elp_params(self):
        
        if self._elp_params['orientation'] > 90 or self._elp_params['orientation'] < 0:
            raise ValueError('Orientation must be between 0 and 90 degrees')
        
        self._amplitude = self._elp_params['axis_ratio']
        self._orientation = self._elp_params['orientation']

    def correct_edp(self, edp: eDiffractionPattern):
        stretch_factor = (1 + self._amplitude) / 2
        orientation = self._orientation

        print(stretch_factor, orientation)

        img_padder = ImagePadder(edp.data, edp.center)

        data = img_padder.square_padded_data

        pivot_point = np.array(data.shape) // 2

        correction_pipeline: List[CorrectionOperation] = []

        add_border = AddBorder(data.shape)
        correction_pipeline.append(add_border)

        first_rotation = Rotate(orientation, pivot_point + np.array(2 * [add_border.border_size]))
        correction_pipeline.append(first_rotation)

        first_scale = Scale(stretch_factor)
        correction_pipeline.append(first_scale)

        second_rotation = Rotate(90, pivot_point + np.array(2 * [add_border.border_size]))
        correction_pipeline.append(second_rotation)

        second_scale = Scale(1 / stretch_factor)
        correction_pipeline.append(second_scale)

        final_rotation = Rotate(-(orientation+90), pivot_point + np.array(2 * [add_border.border_size]))
        correction_pipeline.append(final_rotation)

        remove_border = RemoveBorder(add_border.border_size)
        correction_pipeline.append(remove_border)

        for operation in correction_pipeline:
            data = operation.operate(data).copy()

        
        
        data = img_padder.recover_original_shape(data)

        return data

        