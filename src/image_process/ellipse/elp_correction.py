import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

from src.image_process.utils import ImagePadder
from src.image_process.diffraction_pattern import eDiffractionPattern


class TransformationStep(ABC):
    """
    Abstract base class for transformation steps in an image processing pipeline.
    """

    @abstractmethod
    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the transformation step on the input data.

        Args:
            data (np.ndarray): The input image data.

        Returns:
            np.ndarray: The transformed image data.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        raise NotImplementedError


class Pad(TransformationStep):
    """
    A transformation step that pads the image to make it square, centered around a specified point.
    """

    _shared_img_padder = None

    def __init__(self, center: Tuple[int, int]):
        """
        Initializes the Pad transformation step.

        Args:
            center (Tuple[int, int]): The (x, y) coordinates of the center point for padding.

        Raises:
            TypeError: If `center` is not a tuple of two integers.
        """
        if not isinstance(center, tuple) or len(center) != 2 or not all(isinstance(c, int) for c in center):
            raise TypeError("Input `center` must be a tuple of two integers.")
        self.center = center

    @classmethod
    def get_img_padder(cls) -> ImagePadder:
        """
        Returns the shared ImagePadder instance.

        Returns:
            ImagePadder: The shared ImagePadder instance.

        Raises:
            ValueError: If the ImagePadder instance is not initialized.
        """
        if cls._shared_img_padder is None:
            raise ValueError("ImagePadder is not initialized.")
        return cls._shared_img_padder

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the padding transformation.

        Args:
            data (np.ndarray): The input image data.

        Returns:
            np.ndarray: The padded image data.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        Pad._shared_img_padder = ImagePadder(data, self.center)
        return Pad._shared_img_padder.square_padded_data


class ReversePad(TransformationStep):
    """
    A transformation step that reverses the padding applied by the Pad step.
    """

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the reverse padding transformation.

        Args:
            data (np.ndarray): The padded image data.

        Returns:
            np.ndarray: The image data with padding removed.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        return Pad.get_img_padder().recover_original_shape(data)


class AffineTransformation(TransformationStep):
    """
    A transformation step that applies an affine transformation to the image.
    """

    def __init__(self, matrix: np.ndarray):
        """
        Initializes the AffineTransformation step.

        Args:
            matrix (np.ndarray): The 3x3 affine transformation matrix.

        Raises:
            TypeError: If `matrix` is not a numpy array.
            ValueError: If `matrix` is not a 3x3 array.
        """
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input `matrix` must be a numpy array.")
        if matrix.shape != (3, 3):
            raise ValueError("Input `matrix` must be a 3x3 array.")
        self._matrix = matrix

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Executes the affine transformation.

        Args:
            data (np.ndarray): The input image data.

        Returns:
            np.ndarray: The transformed image data.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        height, width = data.shape
        transformed_image = cv2.warpAffine(data, self.matrix[:-1, :], (width, height))
        return transformed_image

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns the affine transformation matrix.

        Returns:
            np.ndarray: The 3x3 affine transformation matrix.

        Raises:
            ValueError: If the matrix is not initialized.
        """
        if self._matrix is None:
            raise ValueError("Matrix is not initialized.")
        return self._matrix


class Rotate(AffineTransformation):
    """
    A transformation step that rotates the image by a specified angle.
    """

    def __init__(self, angle: float):
        """
        Initializes the Rotate transformation step.

        Args:
            angle (float): The rotation angle in degrees.

        Raises:
            TypeError: If `angle` is not a float or integer.
        """
        if not isinstance(angle, (float, int)):
            raise TypeError("Input `angle` must be a float or integer.")
        rotation_matrix = Rotate.compute_rotation_matrix(angle)
        super().__init__(rotation_matrix)

    @staticmethod
    def compute_rotation_matrix(angle: float) -> np.ndarray:
        """
        Computes the 3x3 rotation matrix for a given angle.

        Args:
            angle (float): The rotation angle in degrees.

        Returns:
            np.ndarray: The 3x3 rotation matrix.
        """
        rotation_angle = np.radians(angle)
        rotation_matrix = np.array(
            [
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle), np.cos(rotation_angle), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return rotation_matrix


class Scale(AffineTransformation):
    """
    A transformation step that scales the image along the x and y axes.
    """

    def __init__(self, axis_ratio: float):
        """
        Initializes the Scale transformation step.

        Args:
            axis_ratio (float): The ratio of the major axis to the minor axis.

        Raises:
            TypeError: If `axis_ratio` is not a float or integer.
            ValueError: If `axis_ratio` is not positive.
        """
        if not isinstance(axis_ratio, (float, int)):
            raise TypeError("Input `axis_ratio` must be a float or integer.")
        if axis_ratio <= 0:
            raise ValueError("Input `axis_ratio` must be positive.")

        scale_x = np.sqrt(axis_ratio)
        scale_y = 1 / np.sqrt(axis_ratio)
        scaling_matrix = Scale.compute_scale_matrix(scale_x, scale_y)
        super().__init__(scaling_matrix)

    @staticmethod
    def compute_scale_matrix(scale_x: float, scale_y: float) -> np.ndarray:
        """
        Computes the 3x3 scaling matrix for given x and y scaling factors.

        Args:
            scale_x (float): The scaling factor along the x-axis.
            scale_y (float): The scaling factor along the y-axis.

        Returns:
            np.ndarray: The 3x3 scaling matrix.
        """
        scaling_matrix = np.array(
            [
                [scale_x, 0, 0],
                [0, scale_y, 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        return scaling_matrix


class ComposedTransform(AffineTransformation):
    """
    A transformation step that combines multiple affine transformations into a single transformation.
    """

    def __init__(self, affine_transformations: List[AffineTransformation]):
        """
        Initializes the ComposedTransform step.

        Args:
            affine_transformations (List[AffineTransformation]): A list of affine transformations to combine.

        Raises:
            TypeError: If `affine_transformations` is not a list of AffineTransformation instances.
        """
        if not isinstance(affine_transformations, list) or not all(
            isinstance(t, AffineTransformation) for t in affine_transformations
        ):
            raise TypeError("Input `affine_transformations` must be a list of AffineTransformation instances.")

        combined_matrix = ComposedTransform._combine_transformations(affine_transformations)
        super().__init__(combined_matrix)

    @staticmethod
    def _combine_transformations(transformations: List[AffineTransformation]) -> np.ndarray:
        """
        Combines multiple affine transformations into a single transformation matrix.

        Args:
            transformations (List[AffineTransformation]): A list of affine transformations.

        Returns:
            np.ndarray: The combined 3x3 transformation matrix.
        """
        combined_matrix = np.eye(3)
        for transformation in transformations:
            combined_matrix = combined_matrix @ transformation.matrix
        return combined_matrix


class CorrectionPipeline:
    """
    A pipeline for applying a sequence of transformation steps to an image.
    """

    def __init__(self):
        """Initializes the CorrectionPipeline."""
        self.steps: List[TransformationStep] = []

    def add_step(self, step: TransformationStep):
        """
        Adds a transformation step to the pipeline.

        Args:
            step (TransformationStep): The transformation step to add.

        Raises:
            TypeError: If `step` is not an instance of TransformationStep.
        """
        if not isinstance(step, TransformationStep):
            raise TypeError("Input `step` must be an instance of TransformationStep.")
        self.steps.append(step)

    def execute(self, data: np.ndarray) -> np.ndarray:
        """
        Executes all transformation steps in the pipeline on the input data.

        Args:
            data (np.ndarray): The input image data.

        Returns:
            np.ndarray: The transformed image data.

        Raises:
            TypeError: If `data` is not a numpy array.
            ValueError: If `data` is not a 2D array.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Input `data` must be a numpy array.")
        if len(data.shape) != 2:
            raise ValueError("Input `data` must be a 2D array.")

        for step in self.steps:
            data = step.execute(data)
        return data


def correct_ellipse(edp: eDiffractionPattern, ellipse_params: Dict[str, float]) -> np.ndarray:
    """
    Corrects the elliptical distortion in the diffraction pattern using a pipeline of transformations.

    Args:
        edp (eDiffractionPattern): The diffraction pattern to correct.
        ellipse_params (Dict[str, float]): A dictionary containing the ellipse parameters:
                                           - 'orientation': The orientation angle of the ellipse.
                                           - 'axis_ratio': The ratio of the major axis to the minor axis.

    Returns:
        np.ndarray: The corrected diffraction pattern.

    Raises:
        TypeError: If `edp` is not an instance of eDiffractionPattern or if `ellipse_params` is not a dictionary.
        ValueError: If `ellipse_params` is missing required keys or contains invalid values.
    """
    if not isinstance(edp, eDiffractionPattern):
        raise TypeError("Input `edp` must be an instance of eDiffractionPattern.")
    if not isinstance(ellipse_params, dict):
        raise TypeError("Input `ellipse_params` must be a dictionary.")
    if "orientation" not in ellipse_params or "axis_ratio" not in ellipse_params:
        raise ValueError("Input `ellipse_params` must contain 'orientation' and 'axis_ratio' keys.")
    if not isinstance(ellipse_params["orientation"], (float, int)) or not isinstance(
        ellipse_params["axis_ratio"], (float, int)
    ):
        raise ValueError("Values in `ellipse_params` must be floats or integers.")

    pipeline = CorrectionPipeline()

    pipeline.add_step(Pad(edp.center))

    affine_transformations = []

    rotation = Rotate(ellipse_params["orientation"])
    affine_transformations.append(rotation)

    scaling = Scale(ellipse_params["axis_ratio"])
    affine_transformations.append(scaling)

    inverse_rotation = Rotate(-ellipse_params["orientation"])
    affine_transformations.append(inverse_rotation)

    pipeline.add_step(ComposedTransform(affine_transformations))

    pipeline.add_step(ReversePad())

    result = pipeline.execute(edp.data)
    return result