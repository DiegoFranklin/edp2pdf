# File: tests/image_process/mask/test_mask_utils.py
import pytest
import numpy as np
from edp2pdf.image_process.mask.utils import (
    expand_hole,
    compute_cyclic_shift,
    find_ones_group_limits,
    get_valid_theta_range,
)

# Fixtures
@pytest.fixture
def angular_data():
    theta = np.linspace(0, 360, 100, endpoint=False)
    mask = np.zeros(100, dtype=np.uint8)
    mask[20:80] = 1  # Simulate a valid angular range
    return mask, theta

# Tests for utils functions
def test_expand_hole(angular_data):
    precursor_mask, theta = angular_data
    expanded_mask = expand_hole(precursor_mask, theta, angular_range_expansion=30)
    assert isinstance(expanded_mask, np.ndarray)
    assert expanded_mask.shape == precursor_mask.shape

    # Test invalid inputs
    with pytest.raises(TypeError):
        expand_hole("invalid_mask", theta, 30)
    with pytest.raises(ValueError):
        expand_hole(precursor_mask, theta, angular_range_expansion=400)

def test_compute_cyclic_shift(angular_data):
    precursor_mask, theta = angular_data
    shifted_mask = compute_cyclic_shift(precursor_mask, theta, cyclic_shift=90)
    assert isinstance(shifted_mask, np.ndarray)
    assert shifted_mask.shape == precursor_mask.shape

    # Test invalid inputs
    with pytest.raises(TypeError):
        compute_cyclic_shift("invalid_mask", theta, 90)

def test_find_ones_group_limits():
    arr = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0])
    start, end = find_ones_group_limits(arr)
    assert start == 2
    assert end == 5

    # Test invalid inputs
    with pytest.raises(TypeError):
        find_ones_group_limits("invalid_array")
    with pytest.raises(ValueError):
        find_ones_group_limits(np.array([0, 0, 0]))

def test_get_valid_theta_range(angular_data):
    angular_mask, theta_space = angular_data
    start, end = get_valid_theta_range(angular_mask, theta_space)
    assert isinstance(start, float)
    assert isinstance(end, float)

    # Test invalid inputs
    with pytest.raises(TypeError):
        get_valid_theta_range("invalid_mask", theta_space)
    with pytest.raises(ValueError):
        get_valid_theta_range(np.zeros(100), theta_space)