import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

from src.image_process.utils import ImagePadder
from src.image_process.diffraction_pattern import eDiffractionPattern

class TransformationStep(ABC):
    def execute(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Pad(TransformationStep):
    _shared_img_padder = None
    
    def __init__(self, center: Tuple[int, int]):
        self.center = center

    @classmethod
    def get_img_padder(cls) -> ImagePadder:
        return cls._shared_img_padder

    def execute(self, data: np.ndarray) -> np.ndarray:
        Pad._shared_img_padder = ImagePadder(data, self.center)
        return Pad._shared_img_padder.square_padded_data
    
class ReversePad(TransformationStep):
    def execute(self, data: np.ndarray) -> np.ndarray:
        return Pad.get_img_padder().recover_original_shape(data)

class AffineTransformation(TransformationStep):
    def __init__(self, matrix: np.ndarray):
        self._matrix = matrix 

    def execute(self, data: np.ndarray) -> np.ndarray:
        height, width = data.shape
        transformed_image = cv2.warpAffine(data, self.matrix[:-1,:], (width, height))
        return transformed_image
    
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise ValueError("Matrix is not initialized.")
        return self._matrix

class Rotate(AffineTransformation):
    def __init__(self, angle: float):
        rotation_matrix = Rotate.compute_rotation_matrix(angle)
        super().__init__(rotation_matrix)
    
    @staticmethod
    def compute_rotation_matrix(angle: float) -> np.ndarray:
        rotation_angle = np.radians(angle)
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle),0],
                                    [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                                    [0,                      0,                      1]], dtype=np.float32)
        
        return rotation_matrix

class Scale(AffineTransformation):
    def __init__(self, axis_ratio: float):
        scale_x = np.sqrt(axis_ratio)
        scale_y = 1 / np.sqrt(axis_ratio)

        scaling_matrix = Scale.compute_scale_matrix(scale_x, scale_y)
        super().__init__(scaling_matrix)
    
    @staticmethod
    def compute_scale_matrix(scale_x: float, scale_y: float) -> np.ndarray:
        scaling_matrix = np.array([[scale_x, 0, 0],
                                   [0, scale_y, 0],
                                   [0, 0,       1]], dtype=np.float32)
        
        return scaling_matrix


class ComposedTransform(AffineTransformation):
    def __init__(self, affine_transformations: List[AffineTransformation]):
    
        combined_matrix = ComposedTransform._combine_transformations(affine_transformations)

        super().__init__(combined_matrix)

    @staticmethod
    def _combine_transformations(transformations: List[AffineTransformation]):
        combined_matrix = np.eye(3)
        for transformation in transformations:
            combined_matrix = combined_matrix @ transformation.matrix
        return combined_matrix

class CorrectionPipeline:
    def __init__(self):
        self.steps: List[TransformationStep] = []

    def add_step(self, step: TransformationStep):
        self.steps.append(step)
    
    def execute(self, data: np.ndarray) -> np.ndarray:
        for step in self.steps:
            data = step.execute(data)
        return data
    
def correct_ellipse(edp: eDiffractionPattern, ellipse_params: Dict[str, float]) -> np.ndarray:

    pipeline = CorrectionPipeline()

    pipeline.add_step(Pad(edp.center))

    affine_transformations = []

    rotation = Rotate(ellipse_params['orientation'])
    affine_transformations.append(rotation)

    scaling = Scale(ellipse_params['axis_ratio'])
    affine_transformations.append(scaling)

    inverse_rotation = Rotate(-ellipse_params['orientation'])
    affine_transformations.append(inverse_rotation)

    pipeline.add_step(ComposedTransform(affine_transformations))


    pipeline.add_step(ReversePad())
    
    result = pipeline.execute(edp.data)
    return result
