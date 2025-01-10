from edp2pdf.image_process.polar.polar_representation import PolarRepresentation
from edp2pdf.image_process.polar.polar_transformation import CVPolarTransformation, PolarTransformation

import numpy as np
import cv2
import pytest

class TestCVPolarTransformation:

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up the test case with a sample image."""
        self.transformation = CVPolarTransformation(interpolation_method=cv2.INTER_CUBIC)

        self.test_image = np.zeros((2048, 2048), dtype=np.uint8)
        self.center = (1024, 1024)
        self.test_image[300:1500, 300:1500] = 255
        self.polar_image = self.transformation.transform(self.test_image, self.center)

    def test_transform_basic(self):
        """Test if the transform method works and returns a valid output."""
        polar_image = self.transformation.transform(self.test_image, self.center)
        assert isinstance(polar_image, np.ndarray), "Output should be a numpy array"

    def test_transform_shape(self):
        """Test if the output shape is correct."""
        # Calculate expected dimensions

        max_radius = self.transformation._max_radius
        
        assert(np.pi * max_radius ** 2 >= np.multiply(*self.test_image.shape), "Output shape does not match expected dimensions")

    def test_transform_with_different_interpolation(self):
        """Test transformation with different interpolation methods."""
        transformation_linear = CVPolarTransformation(interpolation_method=cv2.INTER_LINEAR)
        polar_image_linear = transformation_linear.transform(self.test_image, self.center)

        transformation_nearest = CVPolarTransformation(interpolation_method=cv2.INTER_NEAREST)
        polar_image_nearest = transformation_nearest.transform(self.test_image, self.center)

        assert isinstance(polar_image_linear, np.ndarray), "Output should be a numpy array for INTER_LINEAR"
        assert isinstance(polar_image_nearest, np.ndarray), "Output should be a numpy array for INTER_NEAREST"


    def test_transform_center_out_of_bounds(self):
        """Test transformation with a center point out of the image bounds."""
        out_of_bounds_center = (15000, 15000)  # Center point outside the image
        with pytest.raises(IndexError):
            self.transformation.transform(self.test_image, out_of_bounds_center)
        out_of_bounds_center = (-15000, -15000)  # Center point outside the image
        with pytest.raises(IndexError):
            self.transformation.transform(self.test_image, out_of_bounds_center)