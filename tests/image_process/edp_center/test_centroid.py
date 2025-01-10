import pytest
import numpy as np
from edp2pdf.image_process.edp_center.centroid import get_centroid

@pytest.fixture
def setup_image():
    return np.random.rand(100, 100)

def test_get_centroid(setup_image):
    image = setup_image
    centroid = get_centroid(image)

    # Test return type
    assert isinstance(centroid, tuple)

    # Test centroid coordinates are within bounds
    assert 0 <= centroid[0] < image.shape[1]
    assert 0 <= centroid[1] < image.shape[0]

    # Test invalid input type
    with pytest.raises(TypeError):
        get_centroid([1, 2, 3])

    # Test invalid input shape
    with pytest.raises(ValueError):
        get_centroid(np.random.rand(100, 100, 3))

    # Test zero mass
    with pytest.raises(ValueError):
        get_centroid(np.zeros((100, 100)))