import cv2
import numpy as np
from abc import ABC
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
        self.matrix = matrix  # Matrix is publicly accessible now

    def execute(self, data: np.ndarray) -> np.ndarray:
        height, width = data.shape
        transformed_image = cv2.warpAffine(data, self.matrix, (width, height))
        return transformed_image

class Rotate(AffineTransformation):
    def __init__(self, angle: float, center: Tuple[int, int]):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        super().__init__(rotation_matrix)

class Scale(AffineTransformation):
    def __init__(self, axis_ratio: float, center: Tuple[int, int]):
        scale_x = np.sqrt(axis_ratio)
        scale_y = 1 / np.sqrt(axis_ratio)
        scaling_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
        super().__init__(scaling_matrix)

class CompositeTransform(AffineTransformation):
    def __init__(self, center: Tuple[int, int], angle: float, axis_ratio: float):
        # Initial rotation
        rotation_matrix_1 = cv2.getRotationMatrix2D(center, angle, 1)
        
        # Scaling
        scale_x = np.sqrt(axis_ratio)
        scale_y = 1 / np.sqrt(axis_ratio)
        scaling_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
        
        # Inverse rotation
        rotation_matrix_2 = cv2.getRotationMatrix2D(center, -angle, 1)
        
        # Combine transformations
        combined_matrix = np.dot(rotation_matrix_2, np.dot(scaling_matrix, rotation_matrix_1))
        
        super().__init__(combined_matrix)

class CorrectionPipeline:
    def __init__(self):
        self.steps: List[TransformationStep] = []

    def add_step(self, step: TransformationStep):
        self.steps.append(step)
    
    def execute(self, data: np.ndarray) -> np.ndarray:
        for step in self.steps:
            data = step.execute(data)
        return data

    def get_transformation_matrices(self) -> List[np.ndarray]:
        """Returns a list of transformation matrices for each affine step."""
        return [step.matrix for step in self.steps if isinstance(step, AffineTransformation)]
    
def correct_ellipse(edp: eDiffractionPattern, ellipse_params: Dict[str, float]) -> np.ndarray:
    pipeline = CorrectionPipeline()

    # Step 1: Pad
    pipeline.add_step(Pad(edp.center))

    # Step 2: Composite Transform (rotation + scaling + inverse rotation)
    pipeline.add_step(CompositeTransform(edp.center, ellipse_params['orientation'], ellipse_params['axis_ratio']))

    # Step 3: Reverse Pad
    pipeline.add_step(ReversePad())
    
    result = pipeline.execute(edp.data)
    return result
