import pytest
import numpy as np
from edp2pdf.image_process.edp_center.first_estimation.autocorrelation import autocorrelation
import cv2

@pytest.fixture
def setup_image():
    flat = np.zeros((512, 512))
    img = cv2.circle(flat, (256, 256), 100, 1, -1)
    return img


def test_autocorrelation(setup_image):
    image = setup_image
    center = autocorrelation(image, mask=np.ones_like(image))

    # Test return type
    assert isinstance(center, tuple)

    # Test centroid coordinates are within bounds
    assert 0 <= center[0] < image.shape[1]
    assert 0 <= center[1] < image.shape[0]
