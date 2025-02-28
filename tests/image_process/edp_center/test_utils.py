import pytest
import numpy as np
from edp2pdf.image_process.edp_center.utils import (
    get_centered_crop_img,
    bilinear_interpolation,
    get_integer_neighbors,
)

@pytest.fixture
def setup_image_and_center():
    image = np.random.rand(100, 100)
    center = (50, 50)
    return image, center

@pytest.fixture
def set_complex_image_and_center():
    image = np.random.rand(2048, 2048)
    max_radius = 1000

    center = (max_radius, 1024)

    return image, center, max_radius

def test_get_centered_crop_img(setup_image_and_center, set_complex_image_and_center):
    image, center = setup_image_and_center
    cropped = get_centered_crop_img(image, center)

    # Test return type
    assert isinstance(cropped, np.ndarray)

    # Test cropped image shape
    assert cropped.shape == (100, 100)

    # Test invalid input type
    with pytest.raises(TypeError):
        get_centered_crop_img([1, 2, 3], center)

    # Test invalid center type
    with pytest.raises(TypeError):
        get_centered_crop_img(image, [50, 50])

    # Test center out of bounds
    with pytest.raises(ValueError):
        get_centered_crop_img(image, (150, 150))

    image, center, max_radius = set_complex_image_and_center

    cropped = get_centered_crop_img(image, center)

    # Test cropped image shape
    assert cropped.shape[0] == cropped.shape[1]
    assert cropped.shape == (2 * max_radius, 2 * max_radius)

def test_bilinear_interpolation():
    point = (1.5, 1.5)
    neighbors = ((1, 1), (1, 2), (2, 1), (2, 2))
    values = (1.0, 2.0, 3.0, 4.0)

    interpolated_value = bilinear_interpolation(point, neighbors, values)

    # Test return type
    assert isinstance(interpolated_value, float)

    # Test expected interpolation result
    expected_value = 2.5  # (1 + 2 + 3 + 4) / 4
    assert interpolated_value == expected_value

    # Test invalid point type
    with pytest.raises(TypeError):
        bilinear_interpolation([1.5, 1.5], neighbors, values)

    # Test invalid neighbors type
    with pytest.raises(TypeError):
        bilinear_interpolation(point, [(1, 1), (1, 2), (2, 1), (2, 2)], values)


def test_get_integer_neighbors():
    point = (1.5, 1.5)
    neighbors = get_integer_neighbors(point)

    # Test return type
    assert isinstance(neighbors, tuple)

    # Test expected neighbors
    expected_neighbors = ((1, 1), (1, 2), (2, 1), (2, 2))
    assert neighbors == expected_neighbors

    # Test invalid point type
    with pytest.raises(TypeError):
        get_integer_neighbors([1.5, 1.5])