import pytest
import numpy as np
from edp2pdf.image_process.edp_center.center_optimization.opt_funcs import Distance

@pytest.fixture
def setup_data_and_mask():
    data = np.random.rand(100, 100)
    mask = np.random.randint(0, 2, (100, 100))
    return data, mask

def test_distance_initialization(setup_data_and_mask):
    data, mask = setup_data_and_mask

    # Test valid initialization
    distance = Distance(data, mask)
    assert isinstance(distance, Distance)

    # Test invalid data type
    with pytest.raises(TypeError):
        Distance([1, 2, 3], mask)

    # Test invalid mask type
    with pytest.raises(TypeError):
        Distance(data, [1, 2, 3])

    # Test shape mismatch
    with pytest.raises(ValueError):
        Distance(data, np.random.rand(50, 50))

def test_distance_get_point_evaluation(setup_data_and_mask):
    data, mask = setup_data_and_mask
    distance = Distance(data, mask)

    # Test valid point evaluation
    point = (50, 50)
    penalty = distance._get_point_evaluation(point)
    assert isinstance(penalty, float)

    # Test invalid point type
    with pytest.raises(ValueError):
        distance._get_point_evaluation((50.5, 50.5))

    # Test point out of bounds
    with pytest.raises(ValueError):
        distance._get_point_evaluation((150, 150))