import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

from src.image_process.utils import ImagePadder
from src.image_process.diffraction_pattern import eDiffractionPattern

class TransformationStep(ABC):
    @abstractmethod
    def execute(self, data: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
class AffineTransformation(TransformationStep):
    def __init__(self, matrix: np.ndarray):
        self.matrix = matrix

    def execute(self, data: np.ndarray) -> np.ndarray:
        height, width = data.shape
        transformed_image = cv2.warpAffine(data, self.matrix, (width, height))
        return transformed_image

class Rotate(AffineTransformation):
    def __init__(self, angle: float, center: Tuple[int, int]):
        # Initialize with a rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        super().__init__(rotation_matrix)

class Scale(AffineTransformation):
    def __init__(self, axis_ratio: float, center: Tuple[int, int]):
        # Scaling matrix
        scale_x = np.sqrt(axis_ratio)
        scale_y = 1 / np.sqrt(axis_ratio)
        scaling_matrix = np.array([[scale_x, 0, 0], [0, scale_y, 0]], dtype=np.float32)
        super().__init__(scaling_matrix)

class Pad(TransformationStep):
    _shared_img_padder = None
    
    def __init__(self, center: Tuple[int, int]):
        self.center = center

    @classmethod
    def get_img_padder(cls) -> ImagePadder:
        return cls._shared_img_padder

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Pads the given data to create a square image centered on the
        specified center.

        This method first checks if the shared ImagePadder has been created
        (i.e., if the execute method has been called previously). If not, it
        creates the shared ImagePadder and uses it to pad the data. The
        padded data is then returned.

        Parameters
        ----------
        data : np.ndarray
            The data to be padded.

        Returns
        -------
        np.ndarray
            The padded data.
        """

        Pad._shared_img_padder = ImagePadder(data, self.center)
        return Pad._shared_img_padder.square_padded_data
    
class ReversePad(TransformationStep):
    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Recovers the original shape of the data from the padded data.
        
        Parameters
        ----------
        data : np.ndarray
            The padded data.
        
        Returns
        -------
        np.ndarray
            The original data, with the same shape as the input data to the
            constructor.
        """
        
        return Pad.get_img_padder().recover_original_shape(data)



class CorrectionPipeline:
    def __init__(self,):
        self.steps: List[TransformationStep] = []

    def add_step(self, step: TransformationStep):
        """
        Add a transformation step to the correction pipeline.

        Parameters
        ----------
        step : TransformationStep
            The transformation step to be added to the pipeline.
        """
        self.steps.append(step)
    
    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Execute the sequence of transformation steps on the input data.

        Parameters
        ----------
        data : np.ndarray
            The input data to be processed.

        Returns
        -------
        np.ndarray
            The processed data after applying all transformation steps in the pipeline.
        """
        for step in self.steps:
            data = step.execute(data)
        return data
    
def correct_ellipse(edp: eDiffractionPattern, ellipse_params: Dict[str, float]) -> np.ndarray:

    pipeline = CorrectionPipeline()


    pipeline.add_step(Pad(edp.center))

    pipeline.add_step(Rotate(ellipse_params['orientation']))

    pipeline.add_step(Scale(ellipse_params['axis_ratio']))

    pipeline.add_step(Rotate(-ellipse_params['orientation']))

    pipeline.add_step(ReversePad())
    

    result = pipeline.execute(edp.data)

    return result


