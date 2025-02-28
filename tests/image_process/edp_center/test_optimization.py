import pytest
import numpy as np
from edp2pdf.image_process.edp_center.center_optimization.optimization import optimize_center

def penalty_func(point):
    return np.sum(point**2)

def test_optimize_center():
    data_shape = (100, 100)
    initial_guess = (50, 50)

    # Test valid optimization
    center = optimize_center(penalty_func, data_shape, initial_guess)
    assert isinstance(center, tuple)
    assert len(center) == 2

    # Test invalid penalty_func type
    with pytest.raises(TypeError):
        optimize_center("not_a_function", data_shape, initial_guess)

    # Test invalid data_shape type
    with pytest.raises(TypeError):
        optimize_center(penalty_func, (100.0, 100.0), initial_guess)

    # Test invalid initial_guess type
    with pytest.raises(TypeError):
        optimize_center(penalty_func, data_shape, [50, 50])

    # Test initial_guess out of bounds
    with pytest.raises(ValueError):
        optimize_center(penalty_func, data_shape, (150, 150))